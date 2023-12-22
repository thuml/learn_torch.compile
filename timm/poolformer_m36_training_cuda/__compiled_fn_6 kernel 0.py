
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_youkaichao/oq/coqpa26gno64lywpu2gjzg7impqfa2qgzglq3m6fzlmzg2arjttw.py
# Source Nodes: [mul, mul_1, mul_2, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, sub, sub_1, sub_2, sub_3, sub_4, x_11, x_12, x_19, x_20, x_27, x_28, x_35, x_36, x_4, x_43], Original ATen: [aten.add, aten.mul, aten.sub]
# mul => mul_2
# mul_1 => mul_8
# mul_2 => mul_11
# mul_3 => mul_17
# mul_4 => mul_20
# mul_5 => mul_26
# mul_6 => mul_29
# mul_7 => mul_35
# mul_8 => mul_38
# mul_9 => mul_44
# sub => sub_1
# sub_1 => sub_4
# sub_2 => sub_7
# sub_3 => sub_10
# sub_4 => sub_13
# x_11 => add_6
# x_12 => add_9
# x_19 => add_13
# x_20 => add_16
# x_27 => add_20
# x_28 => add_23
# x_35 => add_27
# x_36 => add_30
# x_4 => add_2
# x_43 => add_34
triton_poi_fused_add_mul_sub_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: 'i32', 32: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(31, 32))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 96
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr7 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr9 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr11 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr12 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr14 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr15 + (x2), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr16 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr17 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr18 + (x2), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr19 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr20 + (x2), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr21 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr22 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr23 + (x2), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr24 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr25 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 - tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp10 + tmp15
    tmp19 = tmp17 * tmp18
    tmp20 = tmp16 + tmp19
    tmp23 = tmp21 - tmp22
    tmp25 = tmp23 * tmp24
    tmp26 = tmp20 + tmp25
    tmp29 = tmp27 * tmp28
    tmp30 = tmp26 + tmp29
    tmp33 = tmp31 - tmp32
    tmp35 = tmp33 * tmp34
    tmp36 = tmp30 + tmp35
    tmp39 = tmp37 * tmp38
    tmp40 = tmp36 + tmp39
    tmp43 = tmp41 - tmp42
    tmp45 = tmp43 * tmp44
    tmp46 = tmp40 + tmp45
    tmp49 = tmp47 * tmp48
    tmp50 = tmp46 + tmp49
    tl.store(out_ptr0 + (y0 + (3136*x2) + (301056*y1)), tmp10, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (3136*x2) + (301056*y1)), tmp20, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (3136*x2) + (301056*y1)), tmp30, xmask & ymask)
    tl.store(out_ptr3 + (y0 + (3136*x2) + (301056*y1)), tmp40, xmask & ymask)
    tl.store(out_ptr4 + (y0 + (3136*x2) + (301056*y1)), tmp50, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/du/cdudhahzed6zlmzaevcvrok2xvzdinssgwqvrjjpvixpjr3yvbau.py
# Source Nodes: [mul_12, mul_13, mul_14, mul_15, mul_16, mul_17, mul_18, mul_19, mul_20, mul_21, sub_10, sub_6, sub_7, sub_8, sub_9, x_56, x_63, x_64, x_71, x_72, x_79, x_80, x_87, x_88, x_95], Original ATen: [aten.add, aten.mul, aten.sub]
# mul_12 => mul_56
# mul_13 => mul_62
# mul_14 => mul_65
# mul_15 => mul_71
# mul_16 => mul_74
# mul_17 => mul_80
# mul_18 => mul_83
# mul_19 => mul_89
# mul_20 => mul_92
# mul_21 => mul_98
# sub_10 => sub_31
# sub_6 => sub_19
# sub_7 => sub_22
# sub_8 => sub_25
# sub_9 => sub_28
# x_56 => add_44
# x_63 => add_48
# x_64 => add_51
# x_71 => add_55
# x_72 => add_58
# x_79 => add_62
# x_80 => add_65
# x_87 => add_69
# x_88 => add_72
# x_95 => add_76
triton_poi_fused_add_mul_sub_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: 'i32', 32: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(31, 32))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr7 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr9 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr11 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr12 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr14 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr15 + (x2), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr16 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr17 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr18 + (x2), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr19 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr20 + (x2), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr21 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr22 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr23 + (x2), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr24 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr25 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 - tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp10 + tmp15
    tmp19 = tmp17 * tmp18
    tmp20 = tmp16 + tmp19
    tmp23 = tmp21 - tmp22
    tmp25 = tmp23 * tmp24
    tmp26 = tmp20 + tmp25
    tmp29 = tmp27 * tmp28
    tmp30 = tmp26 + tmp29
    tmp33 = tmp31 - tmp32
    tmp35 = tmp33 * tmp34
    tmp36 = tmp30 + tmp35
    tmp39 = tmp37 * tmp38
    tmp40 = tmp36 + tmp39
    tmp43 = tmp41 - tmp42
    tmp45 = tmp43 * tmp44
    tmp46 = tmp40 + tmp45
    tmp49 = tmp47 * tmp48
    tmp50 = tmp46 + tmp49
    tl.store(out_ptr0 + (y0 + (784*x2) + (150528*y1)), tmp10, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (784*x2) + (150528*y1)), tmp20, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (784*x2) + (150528*y1)), tmp30, xmask & ymask)
    tl.store(out_ptr3 + (y0 + (784*x2) + (150528*y1)), tmp40, xmask & ymask)
    tl.store(out_ptr4 + (y0 + (784*x2) + (150528*y1)), tmp50, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zk/czky7naynn4yhvoal7okxgr6bbv2lntltq2dh3lfjxcxdu4u5ewa.py
# Source Nodes: [mul_24, mul_25, mul_26, mul_27, mul_28, mul_29, mul_30, mul_31, mul_32, mul_33, mul_34, mul_35, mul_36, mul_37, mul_38, mul_39, mul_40, mul_41, mul_42, mul_43, mul_44, mul_45, mul_46, mul_47, mul_48, mul_49, mul_50, mul_51, mul_52, mul_53, mul_54, mul_55, mul_56, mul_57, sub_12, sub_13, sub_14, sub_15, sub_16, sub_17, sub_18, sub_19, sub_20, sub_21, sub_22, sub_23, sub_24, sub_25, sub_26, sub_27, sub_28, x_108, x_115, x_116, x_123, x_124, x_131, x_132, x_139, x_140, x_147, x_148, x_155, x_156, x_163, x_164, x_171, x_172, x_179, x_180, x_187, x_188, x_195, x_196, x_203, x_204, x_211, x_212, x_219, x_220, x_227, x_228, x_235, x_236, x_243], Original ATen: [aten.add, aten.mul, aten.sub]
# mul_24 => mul_110
# mul_25 => mul_116
# mul_26 => mul_119
# mul_27 => mul_125
# mul_28 => mul_128
# mul_29 => mul_134
# mul_30 => mul_137
# mul_31 => mul_143
# mul_32 => mul_146
# mul_33 => mul_152
# mul_34 => mul_155
# mul_35 => mul_161
# mul_36 => mul_164
# mul_37 => mul_170
# mul_38 => mul_173
# mul_39 => mul_179
# mul_40 => mul_182
# mul_41 => mul_188
# mul_42 => mul_191
# mul_43 => mul_197
# mul_44 => mul_200
# mul_45 => mul_206
# mul_46 => mul_209
# mul_47 => mul_215
# mul_48 => mul_218
# mul_49 => mul_224
# mul_50 => mul_227
# mul_51 => mul_233
# mul_52 => mul_236
# mul_53 => mul_242
# mul_54 => mul_245
# mul_55 => mul_251
# mul_56 => mul_254
# mul_57 => mul_260
# sub_12 => sub_37
# sub_13 => sub_40
# sub_14 => sub_43
# sub_15 => sub_46
# sub_16 => sub_49
# sub_17 => sub_52
# sub_18 => sub_55
# sub_19 => sub_58
# sub_20 => sub_61
# sub_21 => sub_64
# sub_22 => sub_67
# sub_23 => sub_70
# sub_24 => sub_73
# sub_25 => sub_76
# sub_26 => sub_79
# sub_27 => sub_82
# sub_28 => sub_85
# x_108 => add_86
# x_115 => add_90
# x_116 => add_93
# x_123 => add_97
# x_124 => add_100
# x_131 => add_104
# x_132 => add_107
# x_139 => add_111
# x_140 => add_114
# x_147 => add_118
# x_148 => add_121
# x_155 => add_125
# x_156 => add_128
# x_163 => add_132
# x_164 => add_135
# x_171 => add_139
# x_172 => add_142
# x_179 => add_146
# x_180 => add_149
# x_187 => add_153
# x_188 => add_156
# x_195 => add_160
# x_196 => add_163
# x_203 => add_167
# x_204 => add_170
# x_211 => add_174
# x_212 => add_177
# x_219 => add_181
# x_220 => add_184
# x_227 => add_188
# x_228 => add_191
# x_235 => add_195
# x_236 => add_198
# x_243 => add_202
triton_poi_fused_add_mul_sub_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: '*fp32', 78: '*fp32', 79: '*fp32', 80: '*fp32', 81: '*fp32', 82: '*fp32', 83: '*fp32', 84: '*fp32', 85: '*fp32', 86: '*fp32', 87: '*fp32', 88: '*fp32', 89: '*fp32', 90: '*fp32', 91: '*fp32', 92: '*fp32', 93: '*fp32', 94: '*fp32', 95: '*fp32', 96: '*fp32', 97: '*fp32', 98: '*fp32', 99: '*fp32', 100: '*fp32', 101: '*fp32', 102: '*fp32', 103: 'i32', 104: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(103, 104))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr7 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr9 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr11 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr12 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr14 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr15 + (x2), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr16 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr17 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr18 + (x2), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr19 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr20 + (x2), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr21 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr22 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr23 + (x2), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr24 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr25 + (x2), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr26 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr27 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr28 + (x2), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr29 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr30 + (x2), xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr31 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr32 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr33 + (x2), xmask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr34 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr35 + (x2), xmask, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr36 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr37 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr38 + (x2), xmask, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr39 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr40 + (x2), xmask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr41 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr42 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp84 = tl.load(in_ptr43 + (x2), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr44 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr45 + (x2), xmask, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr46 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp92 = tl.load(in_ptr47 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr48 + (x2), xmask, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr49 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr50 + (x2), xmask, eviction_policy='evict_last')
    tmp101 = tl.load(in_ptr51 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp102 = tl.load(in_ptr52 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp104 = tl.load(in_ptr53 + (x2), xmask, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr54 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr55 + (x2), xmask, eviction_policy='evict_last')
    tmp111 = tl.load(in_ptr56 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp112 = tl.load(in_ptr57 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp114 = tl.load(in_ptr58 + (x2), xmask, eviction_policy='evict_last')
    tmp117 = tl.load(in_ptr59 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp118 = tl.load(in_ptr60 + (x2), xmask, eviction_policy='evict_last')
    tmp121 = tl.load(in_ptr61 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp122 = tl.load(in_ptr62 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp124 = tl.load(in_ptr63 + (x2), xmask, eviction_policy='evict_last')
    tmp127 = tl.load(in_ptr64 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp128 = tl.load(in_ptr65 + (x2), xmask, eviction_policy='evict_last')
    tmp131 = tl.load(in_ptr66 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp132 = tl.load(in_ptr67 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp134 = tl.load(in_ptr68 + (x2), xmask, eviction_policy='evict_last')
    tmp137 = tl.load(in_ptr69 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp138 = tl.load(in_ptr70 + (x2), xmask, eviction_policy='evict_last')
    tmp141 = tl.load(in_ptr71 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp142 = tl.load(in_ptr72 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp144 = tl.load(in_ptr73 + (x2), xmask, eviction_policy='evict_last')
    tmp147 = tl.load(in_ptr74 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp148 = tl.load(in_ptr75 + (x2), xmask, eviction_policy='evict_last')
    tmp151 = tl.load(in_ptr76 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp152 = tl.load(in_ptr77 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp154 = tl.load(in_ptr78 + (x2), xmask, eviction_policy='evict_last')
    tmp157 = tl.load(in_ptr79 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp158 = tl.load(in_ptr80 + (x2), xmask, eviction_policy='evict_last')
    tmp161 = tl.load(in_ptr81 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp162 = tl.load(in_ptr82 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp164 = tl.load(in_ptr83 + (x2), xmask, eviction_policy='evict_last')
    tmp167 = tl.load(in_ptr84 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp168 = tl.load(in_ptr85 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 - tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp10 + tmp15
    tmp19 = tmp17 * tmp18
    tmp20 = tmp16 + tmp19
    tmp23 = tmp21 - tmp22
    tmp25 = tmp23 * tmp24
    tmp26 = tmp20 + tmp25
    tmp29 = tmp27 * tmp28
    tmp30 = tmp26 + tmp29
    tmp33 = tmp31 - tmp32
    tmp35 = tmp33 * tmp34
    tmp36 = tmp30 + tmp35
    tmp39 = tmp37 * tmp38
    tmp40 = tmp36 + tmp39
    tmp43 = tmp41 - tmp42
    tmp45 = tmp43 * tmp44
    tmp46 = tmp40 + tmp45
    tmp49 = tmp47 * tmp48
    tmp50 = tmp46 + tmp49
    tmp53 = tmp51 - tmp52
    tmp55 = tmp53 * tmp54
    tmp56 = tmp50 + tmp55
    tmp59 = tmp57 * tmp58
    tmp60 = tmp56 + tmp59
    tmp63 = tmp61 - tmp62
    tmp65 = tmp63 * tmp64
    tmp66 = tmp60 + tmp65
    tmp69 = tmp67 * tmp68
    tmp70 = tmp66 + tmp69
    tmp73 = tmp71 - tmp72
    tmp75 = tmp73 * tmp74
    tmp76 = tmp70 + tmp75
    tmp79 = tmp77 * tmp78
    tmp80 = tmp76 + tmp79
    tmp83 = tmp81 - tmp82
    tmp85 = tmp83 * tmp84
    tmp86 = tmp80 + tmp85
    tmp89 = tmp87 * tmp88
    tmp90 = tmp86 + tmp89
    tmp93 = tmp91 - tmp92
    tmp95 = tmp93 * tmp94
    tmp96 = tmp90 + tmp95
    tmp99 = tmp97 * tmp98
    tmp100 = tmp96 + tmp99
    tmp103 = tmp101 - tmp102
    tmp105 = tmp103 * tmp104
    tmp106 = tmp100 + tmp105
    tmp109 = tmp107 * tmp108
    tmp110 = tmp106 + tmp109
    tmp113 = tmp111 - tmp112
    tmp115 = tmp113 * tmp114
    tmp116 = tmp110 + tmp115
    tmp119 = tmp117 * tmp118
    tmp120 = tmp116 + tmp119
    tmp123 = tmp121 - tmp122
    tmp125 = tmp123 * tmp124
    tmp126 = tmp120 + tmp125
    tmp129 = tmp127 * tmp128
    tmp130 = tmp126 + tmp129
    tmp133 = tmp131 - tmp132
    tmp135 = tmp133 * tmp134
    tmp136 = tmp130 + tmp135
    tmp139 = tmp137 * tmp138
    tmp140 = tmp136 + tmp139
    tmp143 = tmp141 - tmp142
    tmp145 = tmp143 * tmp144
    tmp146 = tmp140 + tmp145
    tmp149 = tmp147 * tmp148
    tmp150 = tmp146 + tmp149
    tmp153 = tmp151 - tmp152
    tmp155 = tmp153 * tmp154
    tmp156 = tmp150 + tmp155
    tmp159 = tmp157 * tmp158
    tmp160 = tmp156 + tmp159
    tmp163 = tmp161 - tmp162
    tmp165 = tmp163 * tmp164
    tmp166 = tmp160 + tmp165
    tmp169 = tmp167 * tmp168
    tmp170 = tmp166 + tmp169
    tl.store(out_ptr0 + (y0 + (196*x2) + (75264*y1)), tmp10, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (196*x2) + (75264*y1)), tmp20, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (196*x2) + (75264*y1)), tmp30, xmask & ymask)
    tl.store(out_ptr3 + (y0 + (196*x2) + (75264*y1)), tmp40, xmask & ymask)
    tl.store(out_ptr4 + (y0 + (196*x2) + (75264*y1)), tmp50, xmask & ymask)
    tl.store(out_ptr5 + (y0 + (196*x2) + (75264*y1)), tmp60, xmask & ymask)
    tl.store(out_ptr6 + (y0 + (196*x2) + (75264*y1)), tmp70, xmask & ymask)
    tl.store(out_ptr7 + (y0 + (196*x2) + (75264*y1)), tmp80, xmask & ymask)
    tl.store(out_ptr8 + (y0 + (196*x2) + (75264*y1)), tmp90, xmask & ymask)
    tl.store(out_ptr9 + (y0 + (196*x2) + (75264*y1)), tmp100, xmask & ymask)
    tl.store(out_ptr10 + (y0 + (196*x2) + (75264*y1)), tmp110, xmask & ymask)
    tl.store(out_ptr11 + (y0 + (196*x2) + (75264*y1)), tmp120, xmask & ymask)
    tl.store(out_ptr12 + (y0 + (196*x2) + (75264*y1)), tmp130, xmask & ymask)
    tl.store(out_ptr13 + (y0 + (196*x2) + (75264*y1)), tmp140, xmask & ymask)
    tl.store(out_ptr14 + (y0 + (196*x2) + (75264*y1)), tmp150, xmask & ymask)
    tl.store(out_ptr15 + (y0 + (196*x2) + (75264*y1)), tmp160, xmask & ymask)
    tl.store(out_ptr16 + (y0 + (196*x2) + (75264*y1)), tmp170, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lh/clhk5r6aa4kw6rvlx4f3y4i6m2hog5mjp57cijxrk32vtxxqkmgg.py
# Source Nodes: [mul_60, mul_61, mul_62, mul_63, mul_64, mul_65, mul_66, mul_67, mul_68, mul_69, sub_30, sub_31, sub_32, sub_33, sub_34, x_256, x_263, x_264, x_271, x_272, x_279, x_280, x_287, x_288, x_295], Original ATen: [aten.add, aten.mul, aten.sub]
# mul_60 => mul_272
# mul_61 => mul_278
# mul_62 => mul_281
# mul_63 => mul_287
# mul_64 => mul_290
# mul_65 => mul_296
# mul_66 => mul_299
# mul_67 => mul_305
# mul_68 => mul_308
# mul_69 => mul_314
# sub_30 => sub_91
# sub_31 => sub_94
# sub_32 => sub_97
# sub_33 => sub_100
# sub_34 => sub_103
# x_256 => add_212
# x_263 => add_216
# x_264 => add_219
# x_271 => add_223
# x_272 => add_226
# x_279 => add_230
# x_280 => add_233
# x_287 => add_237
# x_288 => add_240
# x_295 => add_244
triton_poi_fused_add_mul_sub_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: 'i32', 32: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(31, 32))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr7 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr9 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr11 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr12 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr14 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr15 + (x2), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr16 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr17 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr18 + (x2), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr19 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr20 + (x2), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr21 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr22 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr23 + (x2), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr24 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr25 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 - tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp10 + tmp15
    tmp19 = tmp17 * tmp18
    tmp20 = tmp16 + tmp19
    tmp23 = tmp21 - tmp22
    tmp25 = tmp23 * tmp24
    tmp26 = tmp20 + tmp25
    tmp29 = tmp27 * tmp28
    tmp30 = tmp26 + tmp29
    tmp33 = tmp31 - tmp32
    tmp35 = tmp33 * tmp34
    tmp36 = tmp30 + tmp35
    tmp39 = tmp37 * tmp38
    tmp40 = tmp36 + tmp39
    tmp43 = tmp41 - tmp42
    tmp45 = tmp43 * tmp44
    tmp46 = tmp40 + tmp45
    tmp49 = tmp47 * tmp48
    tmp50 = tmp46 + tmp49
    tl.store(out_ptr0 + (y0 + (49*x2) + (37632*y1)), tmp10, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (49*x2) + (37632*y1)), tmp20, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (49*x2) + (37632*y1)), tmp30, xmask & ymask)
    tl.store(out_ptr3 + (y0 + (49*x2) + (37632*y1)), tmp40, xmask & ymask)
    tl.store(out_ptr4 + (y0 + (49*x2) + (37632*y1)), tmp50, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bl/cblyfakjqvh4q2o7r7pny3354sdzumtwdf4itmumxxacg4zqtqij.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1000*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/la/cla2ejbj7dszqpdi2ky7wedxxaa45atr733fcn4ueyyu526w72df.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 768.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/os/cosfbzcaxz4bppvjlieki4tuvjr5uklpvxyx7kme3e5slt45wjlh.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4u/c4uyxtarkzkojlhlbe3hmthdabjyojwpft36ibqihq6gyrehphhw.py
# Source Nodes: [], Original ATen: [aten.div, aten.mul]

triton_poi_fused_div_mul_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mul_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex // 49)
    x1 = (xindex // 49) % 768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp1 = 49.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nl/cnldbr4har5nc7b3tcuw22demjslybejf5qmsmf6d67komkk6ttk.py
# Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.sum]

triton_red_fused_div_mul_sum_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_mul_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*(r2 // 49)) + (1536*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (768*r2) + (75264*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 49.0
        tmp2 = tmp0 / tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5m/c5mlfkvbuoqetvfgb4umz7yo4lej6x4aw2czz27m74b77yemyzhu.py
# Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.sum]

triton_per_fused_div_mul_sum_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mul_sum_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/um/cum3l4ioguz2cgxxqbnusynw6iiv7iqtn2cfssvnraok4nwbpctr.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 768
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qw/cqwfzs37t4us4ut4cq27etgy6seltvlemnbuzqnnpu4jhvcktsi7.py
# Source Nodes: [x_298], Original ATen: [aten.gelu, aten.gelu_backward]
# x_298 => add_250, erf_35, mul_321
triton_poi_fused_gelu_gelu_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3072
    y1 = (yindex // 3072)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (3072*x2) + (150528*y1)), xmask, eviction_policy='evict_last')
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zj/czjljwesbbutdprwn724y5jn3yvpesfmiz35ow7c4sa5kpgesmj7.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 3072
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (150528*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gn/cgncinb7zh5i5qxlruoitdfdaumd7bev7u77uzh7k5tmw77tselq.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_per_fused_native_group_norm_backward_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 768
    x1 = (xindex // 768)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (49*x3)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (768*r2) + (37632*x1)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr3 + (x0 + (768*r2) + (37632*x1)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp4 = tmp2 - tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tmp1 + tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5c/c5cea7g5snsaanrdhfpjngt5kqkike2ri4otkmsy5a5qumoihfxb.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_per_fused_native_group_norm_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp7 * tmp1
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mn/cmn2xkopjsqepu52pobyaqiuptfnma7zpvycdqaryxlhtxgia4oy.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_group_norm_backward]

triton_poi_fused_add_div_mul_native_group_norm_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_native_group_norm_backward_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 768)
    y0 = yindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y1), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0 + (768*x2) + (37632*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0 + (768*x2) + (37632*y1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (y1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (y1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr8 + (y1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (y3), None, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp8 = tmp6 - tmp7
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 + tmp10
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp1
    tmp18 = tmp17 * tmp1
    tmp19 = tmp18 * tmp1
    tmp20 = 2.657312925170068e-05
    tmp21 = tmp19 * tmp20
    tmp22 = tmp11 * tmp21
    tmp23 = tmp4 + tmp22
    tmp25 = 49.0
    tmp26 = tmp24 / tmp25
    tmp27 = -tmp21
    tmp28 = tmp27 * tmp13
    tmp29 = tmp12 * tmp1
    tmp30 = tmp29 * tmp20
    tmp31 = tmp28 - tmp30
    tmp32 = tmp23 + tmp31
    tmp33 = tmp26 + tmp32
    tmp34 = tmp33 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp33, xmask)
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zv/czv5tdwrpyh5ijkg6wl2fbiiwvqdkgwe2j2eq2hbsb36f56t6iwx.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_per_fused_native_group_norm_backward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 - tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2b/c2bivojlfsnp3zoytccyldvnvn3qk3wfpamlyq7sdepuopp74j4c.py
# Source Nodes: [sub_35], Original ATen: [aten.mul, aten.sub, aten.sum]
# sub_35 => sub_106
triton_red_fused_mul_sub_sum_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sub_sum_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (37632*(r2 // 49)) + (75264*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (75264*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (768*r2) + (75264*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6e/c6eszi2kljniukdt2mphqstczktgazwuqbj6pn2edy53k6mu66rw.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]

triton_per_fused_avg_pool2d_backward_native_group_norm_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_avg_pool2d_backward_native_group_norm_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 7
    r2 = (rindex // 7)
    x0 = xindex
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + ((7*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(7, 2 + r2))))) + (49*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(7, 2 + r1))))), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + ((7*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(7, 2 + r2))))) + (49*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(7, 2 + r1))))), rmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr0 + ((7*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(7, 2 + r2))))) + (49*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(7, 2 + r1))))), rmask, other=0.0)
    tmp28 = tl.load(in_ptr0 + ((7*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(7, 2 + r2))))) + (49*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(7, 2 + r1))))), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr0 + ((7*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(7, 2 + r2))))) + (49*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(7, 2 + r1))))), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr0 + ((7*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(7, 2 + r2))))) + (49*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(7, 2 + r1))))), rmask, other=0.0)
    tmp48 = tl.load(in_ptr0 + ((7*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(7, 2 + r2))))) + (49*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(7, 2 + r1))))), rmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr0 + ((7*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(7, 2 + r2))))) + (49*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(7, 2 + r1))))), rmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.load(in_ptr0 + ((7*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(7, 2 + r2))))) + (49*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(7, 2 + r1))))), rmask, other=0.0)
    tmp68 = tl.load(in_ptr0 + (r3 + (49*x0)), rmask, other=0.0)
    tmp71 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask, other=0.0)
    tmp1 = ((tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(7, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(7, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(7, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(7, 2 + (tl.math.max(0, (-1) + r1)))))
    tmp2 = tmp0 / tmp1
    tmp3 = tl.math.max(0, (-1) + r2)
    tmp4 = tl.math.min(7, 2 + r2)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.math.max(0, (-1) + r1)
    tmp7 = tl.math.min(7, 2 + r1)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp5 & tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp2, tmp10)
    tmp13 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(7, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(7, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r1)))))
    tmp14 = tmp12 / tmp13
    tmp15 = 1 + (tl.math.max(0, (-1) + r1))
    tmp16 = tmp15 < tmp7
    tmp17 = tmp5 & tmp16
    tmp18 = tmp11 + tmp14
    tmp19 = tl.where(tmp17, tmp18, tmp11)
    tmp21 = ((-1)*(tl.math.min(7, 2 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(7, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(7, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))
    tmp22 = tmp20 / tmp21
    tmp23 = 2 + (tl.math.max(0, (-1) + r1))
    tmp24 = tmp23 < tmp7
    tmp25 = tmp5 & tmp24
    tmp26 = tmp19 + tmp22
    tmp27 = tl.where(tmp25, tmp26, tmp19)
    tmp29 = ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(7, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(7, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r2)))))
    tmp30 = tmp28 / tmp29
    tmp31 = 1 + (tl.math.max(0, (-1) + r2))
    tmp32 = tmp31 < tmp4
    tmp33 = tmp32 & tmp8
    tmp34 = tmp27 + tmp30
    tmp35 = tl.where(tmp33, tmp34, tmp27)
    tmp37 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(7, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r1)))))
    tmp38 = tmp36 / tmp37
    tmp39 = tmp32 & tmp16
    tmp40 = tmp35 + tmp38
    tmp41 = tl.where(tmp39, tmp40, tmp35)
    tmp43 = ((-1)*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(7, 3 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r2))
    tmp44 = tmp42 / tmp43
    tmp45 = tmp32 & tmp24
    tmp46 = tmp41 + tmp44
    tmp47 = tl.where(tmp45, tmp46, tmp41)
    tmp49 = ((-1)*(tl.math.min(7, 2 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(7, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(7, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r2))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))
    tmp50 = tmp48 / tmp49
    tmp51 = 2 + (tl.math.max(0, (-1) + r2))
    tmp52 = tmp51 < tmp4
    tmp53 = tmp52 & tmp8
    tmp54 = tmp47 + tmp50
    tmp55 = tl.where(tmp53, tmp54, tmp47)
    tmp57 = ((-1)*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(7, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1))
    tmp58 = tmp56 / tmp57
    tmp59 = tmp52 & tmp16
    tmp60 = tmp55 + tmp58
    tmp61 = tl.where(tmp59, tmp60, tmp55)
    tmp63 = 1 + ((-1)*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(7, 4 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1)) + (tl.math.max(0, (-1) + r2))
    tmp64 = tmp62 / tmp63
    tmp65 = tmp52 & tmp24
    tmp66 = tmp61 + tmp64
    tmp67 = tl.where(tmp65, tmp66, tmp61)
    tmp69 = -tmp68
    tmp70 = tmp69 + tmp67
    tmp72 = tmp70 * tmp71
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp75 = tl.where(rmask, tmp73, 0)
    tmp76 = tl.sum(tmp75, 1)[:, None]
    tmp77 = tl.broadcast_to(tmp70, [XBLOCK, RBLOCK])
    tmp79 = tl.where(rmask, tmp77, 0)
    tmp80 = tl.sum(tmp79, 1)[:, None]
    tl.store(out_ptr0 + (r3 + (49*x0)), tmp67, rmask)
    tl.store(out_ptr1 + (x0), tmp76, None)
    tl.store(out_ptr2 + (x0), tmp80, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/z2/cz2z3ltfrkrtptru6nvgfbsv6nau4xa2jyoyzegl6m72pb4h62j4.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 37632)
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x3), None)
    tmp10 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = -tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 * tmp7
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 - tmp13
    tmp15 = tmp14 * tmp5
    tmp16 = tmp15 * tmp5
    tmp17 = tmp16 * tmp5
    tmp18 = 2.657312925170068e-05
    tmp19 = tmp17 * tmp18
    tmp20 = tmp9 * tmp19
    tmp21 = tmp8 + tmp20
    tmp22 = -tmp19
    tmp23 = tmp22 * tmp11
    tmp24 = tmp10 * tmp5
    tmp25 = tmp24 * tmp18
    tmp26 = tmp23 - tmp25
    tmp27 = tmp21 + tmp26
    tmp28 = tmp0 + tmp27
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/36/c36rarh4jh4vmubbxdajnvcjpkkwg5z3ymxbefhqrcpayc66w5ar.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (37632*(r2 // 49)) + (75264*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (75264*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e5/ce5sk3buvenw7tomlblc77htjthxdtkguascqbmegqvoazfqmw5v.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]

triton_poi_fused_add_mul_native_group_norm_backward_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_group_norm_backward_21', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 768)
    y0 = yindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y1), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0 + (768*x2) + (37632*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0 + (768*x2) + (37632*y1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (y1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (y1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr8 + (y1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_out_ptr1 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp8 = tmp6 - tmp7
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 + tmp10
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp1
    tmp18 = tmp17 * tmp1
    tmp19 = tmp18 * tmp1
    tmp20 = 2.657312925170068e-05
    tmp21 = tmp19 * tmp20
    tmp22 = tmp11 * tmp21
    tmp23 = tmp4 + tmp22
    tmp25 = -tmp21
    tmp26 = tmp25 * tmp13
    tmp27 = tmp12 * tmp1
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp30 = tmp23 + tmp29
    tmp31 = tmp24 + tmp30
    tmp32 = tmp31 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x2 + (49*y3)), tmp31, xmask)
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qx/cqxx5thwzciptvpzjghacggmwrdgugzcvxrmqcjt3m5ivxrzx6hr.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 37632)
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x3), None)
    tmp10 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = -tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 * tmp7
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 - tmp13
    tmp15 = tmp14 * tmp5
    tmp16 = tmp15 * tmp5
    tmp17 = tmp16 * tmp5
    tmp18 = 2.657312925170068e-05
    tmp19 = tmp17 * tmp18
    tmp20 = tmp9 * tmp19
    tmp21 = tmp8 + tmp20
    tmp22 = -tmp19
    tmp23 = tmp22 * tmp11
    tmp24 = tmp10 * tmp5
    tmp25 = tmp24 * tmp18
    tmp26 = tmp23 - tmp25
    tmp27 = tmp21 + tmp26
    tmp28 = tmp0 + tmp27
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5y/c5yr5t4k75dgs6cwivd7y7b6m727agjzy2uasrzf4jlpgohofmg5.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_per_fused_native_group_norm_backward_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 768
    x1 = (xindex // 768)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (37632*x1)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (768*r2) + (37632*x1)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr3 + (x0 + (768*r2) + (37632*x1)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp4 = tmp2 - tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tmp1 + tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ez/ceznzxluh3aqcbiayyzrebcp7sg6ryg33njkz66ybkikhkho7lpp.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_poi_fused_native_group_norm_backward_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_backward_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (37632*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr5 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr7 + (y1), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr8 + (y1), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr9 + (y1), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp8 = tmp6 - tmp7
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 + tmp10
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp1
    tmp18 = tmp17 * tmp1
    tmp19 = tmp18 * tmp1
    tmp20 = 2.657312925170068e-05
    tmp21 = tmp19 * tmp20
    tmp22 = tmp11 * tmp21
    tmp23 = tmp4 + tmp22
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gw/cgwmejbx5evnkhdv4lvicswitkagr5h6hxlxakrb2e5tmk2plv57.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (768*x2) + (37632*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 * tmp7
    tmp10 = tmp9 * tmp7
    tmp11 = 2.657312925170068e-05
    tmp12 = tmp10 * tmp11
    tmp13 = -tmp12
    tmp14 = tmp13 * tmp3
    tmp15 = tmp2 * tmp7
    tmp16 = tmp15 * tmp11
    tmp17 = tmp14 - tmp16
    tmp18 = tmp1 + tmp17
    tmp19 = tmp0 + tmp18
    tmp21 = tmp19 * tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp19, xmask)
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e7/ce72yiwkshmqmzl5coomwbob3chb3zfsy4b4yfrvqp5purtpd34m.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]

triton_per_fused_avg_pool2d_backward_native_group_norm_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_avg_pool2d_backward_native_group_norm_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 7
    r2 = (rindex // 7)
    x0 = xindex
    r3 = rindex
    x4 = xindex % 768
    x5 = (xindex // 768)
    tmp0 = tl.load(in_ptr0 + ((7*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(7, 2 + r2))))) + (49*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(7, 2 + r1))))), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + ((7*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(7, 2 + r2))))) + (49*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(7, 2 + r1))))), rmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr0 + ((7*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(7, 2 + r2))))) + (49*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(7, 2 + r1))))), rmask, other=0.0)
    tmp28 = tl.load(in_ptr0 + ((7*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(7, 2 + r2))))) + (49*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(7, 2 + r1))))), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr0 + ((7*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(7, 2 + r2))))) + (49*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(7, 2 + r1))))), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr0 + ((7*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(7, 2 + r2))))) + (49*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(7, 2 + r1))))), rmask, other=0.0)
    tmp48 = tl.load(in_ptr0 + ((7*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(7, 2 + r2))))) + (49*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(7, 2 + r1))))), rmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr0 + ((7*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(7, 2 + r2))))) + (49*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(7, 2 + r1))))), rmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.load(in_ptr0 + ((7*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(7, 2 + r2))))) + (49*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(7, 2 + r1))))), rmask, other=0.0)
    tmp68 = tl.load(in_ptr0 + (r3 + (49*x0)), rmask, other=0.0)
    tmp71 = tl.load(in_ptr1 + (x4 + (768*r3) + (37632*x5)), rmask, other=0.0)
    tmp1 = ((tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(7, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(7, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(7, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(7, 2 + (tl.math.max(0, (-1) + r1)))))
    tmp2 = tmp0 / tmp1
    tmp3 = tl.math.max(0, (-1) + r2)
    tmp4 = tl.math.min(7, 2 + r2)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.math.max(0, (-1) + r1)
    tmp7 = tl.math.min(7, 2 + r1)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp5 & tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp2, tmp10)
    tmp13 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(7, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(7, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r1)))))
    tmp14 = tmp12 / tmp13
    tmp15 = 1 + (tl.math.max(0, (-1) + r1))
    tmp16 = tmp15 < tmp7
    tmp17 = tmp5 & tmp16
    tmp18 = tmp11 + tmp14
    tmp19 = tl.where(tmp17, tmp18, tmp11)
    tmp21 = ((-1)*(tl.math.min(7, 2 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(7, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(7, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))
    tmp22 = tmp20 / tmp21
    tmp23 = 2 + (tl.math.max(0, (-1) + r1))
    tmp24 = tmp23 < tmp7
    tmp25 = tmp5 & tmp24
    tmp26 = tmp19 + tmp22
    tmp27 = tl.where(tmp25, tmp26, tmp19)
    tmp29 = ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(7, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(7, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r2)))))
    tmp30 = tmp28 / tmp29
    tmp31 = 1 + (tl.math.max(0, (-1) + r2))
    tmp32 = tmp31 < tmp4
    tmp33 = tmp32 & tmp8
    tmp34 = tmp27 + tmp30
    tmp35 = tl.where(tmp33, tmp34, tmp27)
    tmp37 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(7, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r1)))))
    tmp38 = tmp36 / tmp37
    tmp39 = tmp32 & tmp16
    tmp40 = tmp35 + tmp38
    tmp41 = tl.where(tmp39, tmp40, tmp35)
    tmp43 = ((-1)*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(7, 3 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r2))
    tmp44 = tmp42 / tmp43
    tmp45 = tmp32 & tmp24
    tmp46 = tmp41 + tmp44
    tmp47 = tl.where(tmp45, tmp46, tmp41)
    tmp49 = ((-1)*(tl.math.min(7, 2 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(7, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(7, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r2))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))
    tmp50 = tmp48 / tmp49
    tmp51 = 2 + (tl.math.max(0, (-1) + r2))
    tmp52 = tmp51 < tmp4
    tmp53 = tmp52 & tmp8
    tmp54 = tmp47 + tmp50
    tmp55 = tl.where(tmp53, tmp54, tmp47)
    tmp57 = ((-1)*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(7, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(7, 3 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1))
    tmp58 = tmp56 / tmp57
    tmp59 = tmp52 & tmp16
    tmp60 = tmp55 + tmp58
    tmp61 = tl.where(tmp59, tmp60, tmp55)
    tmp63 = 1 + ((-1)*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(7, 4 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(7, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1)) + (tl.math.max(0, (-1) + r2))
    tmp64 = tmp62 / tmp63
    tmp65 = tmp52 & tmp24
    tmp66 = tmp61 + tmp64
    tmp67 = tl.where(tmp65, tmp66, tmp61)
    tmp69 = -tmp68
    tmp70 = tmp69 + tmp67
    tmp72 = tmp70 * tmp71
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp75 = tl.where(rmask, tmp73, 0)
    tmp76 = tl.sum(tmp75, 1)[:, None]
    tmp77 = tl.broadcast_to(tmp70, [XBLOCK, RBLOCK])
    tmp79 = tl.where(rmask, tmp77, 0)
    tmp80 = tl.sum(tmp79, 1)[:, None]
    tl.store(out_ptr0 + (r3 + (49*x0)), tmp67, rmask)
    tl.store(out_ptr1 + (x0), tmp76, None)
    tl.store(out_ptr2 + (x0), tmp80, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tx/ctxgirqwjijxbyckpr7wjwmije3upaietwce3x3tt2za4ytll2jv.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 768)
    y0 = yindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0 + (768*x2) + (37632*y1)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (y1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (y1), None, eviction_policy='evict_last')
    tmp2 = -tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 * tmp7
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 - tmp13
    tmp15 = tmp14 * tmp5
    tmp16 = tmp15 * tmp5
    tmp17 = tmp16 * tmp5
    tmp18 = 2.657312925170068e-05
    tmp19 = tmp17 * tmp18
    tmp20 = tmp9 * tmp19
    tmp21 = tmp8 + tmp20
    tmp22 = -tmp19
    tmp23 = tmp22 * tmp11
    tmp24 = tmp10 * tmp5
    tmp25 = tmp24 * tmp18
    tmp26 = tmp23 - tmp25
    tmp27 = tmp21 + tmp26
    tmp28 = tmp0 + tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dg/cdgn65gcuntcndq27qifdergkwyphw5y63xrqwb776giodqxnqav.py
# Source Nodes: [], Original ATen: [aten.mul]

triton_poi_fused_mul_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cm/ccmg5qruhnmxx35rzpxyav7rer42ajrdrfkmcji4w3wb7h3q6bul.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (75264*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (384*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ls/clso2thsgeg2rr5cwflgzpetdzg6jdop52kg5uymdqgoehf4ecca.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_per_fused_mul_sum_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fa/cfabv22vg55hmvnuhzm2we3rhdyhmzivfkydyhlln3v6pn533gxn.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f3/cf3i4my7mp2mdzmwsbspk7vyyyd66out7dg6v57os5grejgnoul2.py
# Source Nodes: [x_246], Original ATen: [aten.gelu, aten.gelu_backward]
# x_246 => add_208, erf_29, mul_267
triton_poi_fused_gelu_gelu_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1536
    y1 = (yindex // 1536)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (1536*x2) + (301056*y1)), xmask, eviction_policy='evict_last')
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bs/cbs4o3hqao5itdni4weyaalwvom55kz2aupqubuqf35goa6xhvh7.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (301056*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h7/ch7evlbjauobud2lynuzrxhppwb4vudgfajjdfgqeq42u7ne7zs6.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_red_fused_native_group_norm_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 2
    x1 = (xindex // 2) % 384
    x2 = (xindex // 768)
    tmp5 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (98*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (98*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (384*r3) + (37632*x0) + (75264*x2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr3 + (x1 + (384*r3) + (37632*x0) + (75264*x2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tmp2 - tmp3
        tmp6 = tmp4 * tmp5
        tmp7 = tmp1 + tmp6
        tmp8 = tmp0 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pd/cpdg24micxutxgc44sedcgokyynw4mbljted4qkzvsdavu7ykdgb.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_per_fused_native_group_norm_backward_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (2*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oa/coale5xhjhckzqeb272nhodk2nvrrqtvz6tusgi3xxyczcnzu3bj.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_per_fused_native_group_norm_backward_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o3/co3ozx76zuuajijze5haxtlhy7fvgve5xfzuxlsubfva5izhzdof.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_per_fused_native_group_norm_backward_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp7 * tmp1
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jh/cjhnbmvwrwrees2tgbia2uegigubk2wsh7ylqyfjstzrcx6otpg6.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]

triton_poi_fused_add_mul_native_group_norm_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_group_norm_backward_38', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 384)
    y0 = yindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y1), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (y1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (y1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr8 + (y1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_out_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp8 = tmp6 - tmp7
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 + tmp10
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp1
    tmp18 = tmp17 * tmp1
    tmp19 = tmp18 * tmp1
    tmp20 = 1.328656462585034e-05
    tmp21 = tmp19 * tmp20
    tmp22 = tmp11 * tmp21
    tmp23 = tmp4 + tmp22
    tmp25 = -tmp21
    tmp26 = tmp25 * tmp13
    tmp27 = tmp12 * tmp1
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp30 = tmp23 + tmp29
    tmp31 = tmp24 + tmp30
    tmp32 = tmp31 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x2 + (196*y3)), tmp31, xmask)
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ne/cnels62piogvyj4lijfpplosuuyx3xt52uhwd3kd3nkpz7q7jwfy.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_per_fused_native_group_norm_backward_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 - tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lf/clfdbipqyj6gfc2dgf5h6kntprvxkgvowmh26ulp4usvg27wc62g.py
# Source Nodes: [sub_29], Original ATen: [aten.mul, aten.sub, aten.sum]
# sub_29 => sub_88
triton_red_fused_mul_sub_sum_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sub_sum_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x0) + (75264*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mp/cmpryzter4zygk3chswcyz42rv53fr23jz6wbzonc24iiaocrnqi.py
# Source Nodes: [sub_29], Original ATen: [aten.mul, aten.sub, aten.sum]
# sub_29 => sub_88
triton_per_fused_mul_sub_sum_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sub_sum_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lm/clmenrgcw2jso3vd6pmufngmlfc7j56h4gpfzzylto64tavsswrj.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]

triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 14
    r2 = (rindex // 14)
    x0 = xindex
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + ((14*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + ((14*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr0 + ((14*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr0 + ((14*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr0 + ((14*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr0 + ((14*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, other=0.0)
    tmp48 = tl.load(in_ptr0 + ((14*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr0 + ((14*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.load(in_ptr0 + ((14*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, other=0.0)
    tmp68 = tl.load(in_ptr0 + (r3 + (196*x0)), rmask & xmask, other=0.0)
    tmp71 = tl.load(in_ptr1 + (r3 + (196*x0)), rmask & xmask, other=0.0)
    tmp1 = ((tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1)))))
    tmp2 = tmp0 / tmp1
    tmp3 = tl.math.max(0, (-1) + r2)
    tmp4 = tl.math.min(14, 2 + r2)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.math.max(0, (-1) + r1)
    tmp7 = tl.math.min(14, 2 + r1)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp5 & tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp2, tmp10)
    tmp13 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1)))))
    tmp14 = tmp12 / tmp13
    tmp15 = 1 + (tl.math.max(0, (-1) + r1))
    tmp16 = tmp15 < tmp7
    tmp17 = tmp5 & tmp16
    tmp18 = tmp11 + tmp14
    tmp19 = tl.where(tmp17, tmp18, tmp11)
    tmp21 = ((-1)*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))
    tmp22 = tmp20 / tmp21
    tmp23 = 2 + (tl.math.max(0, (-1) + r1))
    tmp24 = tmp23 < tmp7
    tmp25 = tmp5 & tmp24
    tmp26 = tmp19 + tmp22
    tmp27 = tl.where(tmp25, tmp26, tmp19)
    tmp29 = ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2)))))
    tmp30 = tmp28 / tmp29
    tmp31 = 1 + (tl.math.max(0, (-1) + r2))
    tmp32 = tmp31 < tmp4
    tmp33 = tmp32 & tmp8
    tmp34 = tmp27 + tmp30
    tmp35 = tl.where(tmp33, tmp34, tmp27)
    tmp37 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1)))))
    tmp38 = tmp36 / tmp37
    tmp39 = tmp32 & tmp16
    tmp40 = tmp35 + tmp38
    tmp41 = tl.where(tmp39, tmp40, tmp35)
    tmp43 = ((-1)*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r2))
    tmp44 = tmp42 / tmp43
    tmp45 = tmp32 & tmp24
    tmp46 = tmp41 + tmp44
    tmp47 = tl.where(tmp45, tmp46, tmp41)
    tmp49 = ((-1)*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))
    tmp50 = tmp48 / tmp49
    tmp51 = 2 + (tl.math.max(0, (-1) + r2))
    tmp52 = tmp51 < tmp4
    tmp53 = tmp52 & tmp8
    tmp54 = tmp47 + tmp50
    tmp55 = tl.where(tmp53, tmp54, tmp47)
    tmp57 = ((-1)*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1))
    tmp58 = tmp56 / tmp57
    tmp59 = tmp52 & tmp16
    tmp60 = tmp55 + tmp58
    tmp61 = tl.where(tmp59, tmp60, tmp55)
    tmp63 = 1 + ((-1)*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1)) + (tl.math.max(0, (-1) + r2))
    tmp64 = tmp62 / tmp63
    tmp65 = tmp52 & tmp24
    tmp66 = tmp61 + tmp64
    tmp67 = tl.where(tmp65, tmp66, tmp61)
    tmp69 = -tmp68
    tmp70 = tmp69 + tmp67
    tmp72 = tmp70 * tmp71
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp75 = tl.where(rmask & xmask, tmp73, 0)
    tmp76 = tl.sum(tmp75, 1)[:, None]
    tmp77 = tl.broadcast_to(tmp70, [XBLOCK, RBLOCK])
    tmp79 = tl.where(rmask & xmask, tmp77, 0)
    tmp80 = tl.sum(tmp79, 1)[:, None]
    tl.store(out_ptr0 + (r3 + (196*x0)), tmp67, rmask & xmask)
    tl.store(out_ptr1 + (x0), tmp76, xmask)
    tl.store(out_ptr2 + (x0), tmp80, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s7/cs7ozx5ddkpth6ydcwwkmfhfcorzegh6453f6h3jypl33yx637rw.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 75264)
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x3), None)
    tmp10 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = -tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 * tmp7
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 - tmp13
    tmp15 = tmp14 * tmp5
    tmp16 = tmp15 * tmp5
    tmp17 = tmp16 * tmp5
    tmp18 = 1.328656462585034e-05
    tmp19 = tmp17 * tmp18
    tmp20 = tmp9 * tmp19
    tmp21 = tmp8 + tmp20
    tmp22 = -tmp19
    tmp23 = tmp22 * tmp11
    tmp24 = tmp10 * tmp5
    tmp25 = tmp24 * tmp18
    tmp26 = tmp23 - tmp25
    tmp27 = tmp21 + tmp26
    tmp28 = tmp0 + tmp27
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ba/cbamqs3nlp2wffaiiyo4njkgydoygclspd5ajmln2gxtbe3no64q.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_44', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 75264)
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x3), None)
    tmp10 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = -tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 * tmp7
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 - tmp13
    tmp15 = tmp14 * tmp5
    tmp16 = tmp15 * tmp5
    tmp17 = tmp16 * tmp5
    tmp18 = 1.328656462585034e-05
    tmp19 = tmp17 * tmp18
    tmp20 = tmp9 * tmp19
    tmp21 = tmp8 + tmp20
    tmp22 = -tmp19
    tmp23 = tmp22 * tmp11
    tmp24 = tmp10 * tmp5
    tmp25 = tmp24 * tmp18
    tmp26 = tmp23 - tmp25
    tmp27 = tmp21 + tmp26
    tmp28 = tmp0 + tmp27
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bz/cbzw5gc223t6wqaa7lpk5k674oifbvd5o2yqlq7sbrccesyczecl.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_red_fused_native_group_norm_backward_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_backward_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384) % 2
    x2 = (xindex // 768)
    x4 = (xindex // 384)
    tmp5 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (98*x1) + (196*x0) + (75264*x2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (384*r3) + (37632*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (384*r3) + (37632*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr3 + (x0 + (384*r3) + (37632*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tmp2 - tmp3
        tmp6 = tmp4 * tmp5
        tmp7 = tmp1 + tmp6
        tmp8 = tmp0 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/em/cemk4u6ldtnlcbclkqwkis6l66xuxxnirkonagff5s446vid4rea.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_per_fused_native_group_norm_backward_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 384
    x1 = (xindex // 384)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (768*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ld/cldac5kowd2hhq7wtlcukc4fttj7f3thmhadrinahdpnelrpr2kl.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_poi_fused_native_group_norm_backward_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_backward_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr5 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr7 + (y1), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr8 + (y1), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr9 + (y1), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp8 = tmp6 - tmp7
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 + tmp10
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp1
    tmp18 = tmp17 * tmp1
    tmp19 = tmp18 * tmp1
    tmp20 = 1.328656462585034e-05
    tmp21 = tmp19 * tmp20
    tmp22 = tmp11 * tmp21
    tmp23 = tmp4 + tmp22
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l7/cl75vjl4crvhkxbmx4jhnen5xeni2mziiatxy7y34eowbuym7g2q.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 * tmp7
    tmp10 = tmp9 * tmp7
    tmp11 = 1.328656462585034e-05
    tmp12 = tmp10 * tmp11
    tmp13 = -tmp12
    tmp14 = tmp13 * tmp3
    tmp15 = tmp2 * tmp7
    tmp16 = tmp15 * tmp11
    tmp17 = tmp14 - tmp16
    tmp18 = tmp1 + tmp17
    tmp19 = tmp0 + tmp18
    tmp21 = tmp19 * tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp19, xmask)
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i3/ci3gdnyrefrxzir2ory6qa5gtssjiynbvysop5k47bidqiyi2ckq.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]

triton_per_fused_avg_pool2d_backward_native_group_norm_backward_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_avg_pool2d_backward_native_group_norm_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 14
    r2 = (rindex // 14)
    x0 = xindex
    r3 = rindex
    x4 = xindex % 384
    x5 = (xindex // 384)
    tmp0 = tl.load(in_ptr0 + ((14*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + ((14*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr0 + ((14*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr0 + ((14*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr0 + ((14*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr0 + ((14*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, other=0.0)
    tmp48 = tl.load(in_ptr0 + ((14*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr0 + ((14*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.load(in_ptr0 + ((14*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(14, 2 + r2))))) + (196*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(14, 2 + r1))))), rmask & xmask, other=0.0)
    tmp68 = tl.load(in_ptr0 + (r3 + (196*x0)), rmask & xmask, other=0.0)
    tmp71 = tl.load(in_ptr1 + (x4 + (384*r3) + (75264*x5)), rmask & xmask, other=0.0)
    tmp1 = ((tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1)))))
    tmp2 = tmp0 / tmp1
    tmp3 = tl.math.max(0, (-1) + r2)
    tmp4 = tl.math.min(14, 2 + r2)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.math.max(0, (-1) + r1)
    tmp7 = tl.math.min(14, 2 + r1)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp5 & tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp2, tmp10)
    tmp13 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1)))))
    tmp14 = tmp12 / tmp13
    tmp15 = 1 + (tl.math.max(0, (-1) + r1))
    tmp16 = tmp15 < tmp7
    tmp17 = tmp5 & tmp16
    tmp18 = tmp11 + tmp14
    tmp19 = tl.where(tmp17, tmp18, tmp11)
    tmp21 = ((-1)*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))
    tmp22 = tmp20 / tmp21
    tmp23 = 2 + (tl.math.max(0, (-1) + r1))
    tmp24 = tmp23 < tmp7
    tmp25 = tmp5 & tmp24
    tmp26 = tmp19 + tmp22
    tmp27 = tl.where(tmp25, tmp26, tmp19)
    tmp29 = ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2)))))
    tmp30 = tmp28 / tmp29
    tmp31 = 1 + (tl.math.max(0, (-1) + r2))
    tmp32 = tmp31 < tmp4
    tmp33 = tmp32 & tmp8
    tmp34 = tmp27 + tmp30
    tmp35 = tl.where(tmp33, tmp34, tmp27)
    tmp37 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1)))))
    tmp38 = tmp36 / tmp37
    tmp39 = tmp32 & tmp16
    tmp40 = tmp35 + tmp38
    tmp41 = tl.where(tmp39, tmp40, tmp35)
    tmp43 = ((-1)*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r2))
    tmp44 = tmp42 / tmp43
    tmp45 = tmp32 & tmp24
    tmp46 = tmp41 + tmp44
    tmp47 = tl.where(tmp45, tmp46, tmp41)
    tmp49 = ((-1)*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(14, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))
    tmp50 = tmp48 / tmp49
    tmp51 = 2 + (tl.math.max(0, (-1) + r2))
    tmp52 = tmp51 < tmp4
    tmp53 = tmp52 & tmp8
    tmp54 = tmp47 + tmp50
    tmp55 = tl.where(tmp53, tmp54, tmp47)
    tmp57 = ((-1)*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(14, 3 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1))
    tmp58 = tmp56 / tmp57
    tmp59 = tmp52 & tmp16
    tmp60 = tmp55 + tmp58
    tmp61 = tl.where(tmp59, tmp60, tmp55)
    tmp63 = 1 + ((-1)*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(14, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1)) + (tl.math.max(0, (-1) + r2))
    tmp64 = tmp62 / tmp63
    tmp65 = tmp52 & tmp24
    tmp66 = tmp61 + tmp64
    tmp67 = tl.where(tmp65, tmp66, tmp61)
    tmp69 = -tmp68
    tmp70 = tmp69 + tmp67
    tmp72 = tmp70 * tmp71
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp75 = tl.where(rmask & xmask, tmp73, 0)
    tmp76 = tl.sum(tmp75, 1)[:, None]
    tmp77 = tl.broadcast_to(tmp70, [XBLOCK, RBLOCK])
    tmp79 = tl.where(rmask & xmask, tmp77, 0)
    tmp80 = tl.sum(tmp79, 1)[:, None]
    tl.store(out_ptr0 + (r3 + (196*x0)), tmp67, rmask & xmask)
    tl.store(out_ptr1 + (x0), tmp76, xmask)
    tl.store(out_ptr2 + (x0), tmp80, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ez/cezpqaaaub6z6ndrmvez4mqwx4p4a6rcrho5tdrtorijsmnc5225.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_50', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 384)
    y0 = yindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (y1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (y1), None, eviction_policy='evict_last')
    tmp2 = -tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 * tmp7
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 - tmp13
    tmp15 = tmp14 * tmp5
    tmp16 = tmp15 * tmp5
    tmp17 = tmp16 * tmp5
    tmp18 = 1.328656462585034e-05
    tmp19 = tmp17 * tmp18
    tmp20 = tmp9 * tmp19
    tmp21 = tmp8 + tmp20
    tmp22 = -tmp19
    tmp23 = tmp22 * tmp11
    tmp24 = tmp10 * tmp5
    tmp25 = tmp24 * tmp18
    tmp26 = tmp23 - tmp25
    tmp27 = tmp21 + tmp26
    tmp28 = tmp0 + tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m7/cm7voa2crolszgdi6x6z42y2a3rmu4zwfbisd33q36sbtglgmkil.py
# Source Nodes: [], Original ATen: [aten.mul]

triton_poi_fused_mul_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 192
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mm/cmmdtyadeiv4jquglnvlgx4hpejias6dtkkhxvajjds5pbnpwdnh.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (150528*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vp/cvpwinksg47qf3hhf2y2swudvr4mk4wqodgbevcuttltfqvlybma.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_per_fused_mul_sum_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/it/citpbiqskda6cgap5ywllgjo7boeye6rvx7xc7pinzbk5yullc5g.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yr/cyr7clmcok77v7grncsg6vg6c3dyeybu4q26q7oajgqy3htlfwdz.py
# Source Nodes: [x_98], Original ATen: [aten.gelu, aten.gelu_backward]
# x_98 => add_82, erf_11, mul_105
triton_poi_fused_gelu_gelu_backward_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_55', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (768*x2) + (602112*y1)), xmask, eviction_policy='evict_last')
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5k/c5kik63qzhw6pfdkdporrr2ih4o4g4ekef7yilj7ujavchr2nnqf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (602112*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ds/cdsw4nkn72gvkicfqkljyyd3d7yg25rttcmj62qzzyepgotqb74k.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_red_fused_native_group_norm_backward_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_backward_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10752
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 7
    x1 = (xindex // 7) % 192
    x2 = (xindex // 1344)
    tmp5 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (112*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (112*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (192*r3) + (21504*x0) + (150528*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr3 + (x1 + (192*r3) + (21504*x0) + (150528*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tmp2 - tmp3
        tmp6 = tmp4 * tmp5
        tmp7 = tmp1 + tmp6
        tmp8 = tmp0 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r7/cr7hsqwbglpriq7ay2widn5viaudeucny7wu7vqgr2ubmfg6t4wq.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_per_fused_native_group_norm_backward_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (7*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5a/c5a44r6mbcqzaqcaidixg7wyb3wddvjcjprup2d7abxu4cclaewx.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_per_fused_native_group_norm_backward_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 1536
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (784*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cw/ccwzpvxnpteilxfijbfprsimwdd4ywkkekzon6do44u2g7cwgnrj.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_per_fused_native_group_norm_backward_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp7 * tmp1
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ig/cigpo7z2fro3lypl4pc4otsd7unywukcehgi7sehtd2eo45bzwhf.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]

triton_poi_fused_add_mul_native_group_norm_backward_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_group_norm_backward_61', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 192)
    y0 = yindex % 192
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y1), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (y1), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (y1), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr8 + (y1), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_out_ptr1 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp8 = tmp6 - tmp7
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 + tmp10
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp1
    tmp18 = tmp17 * tmp1
    tmp19 = tmp18 * tmp1
    tmp20 = 6.64328231292517e-06
    tmp21 = tmp19 * tmp20
    tmp22 = tmp11 * tmp21
    tmp23 = tmp4 + tmp22
    tmp25 = -tmp21
    tmp26 = tmp25 * tmp13
    tmp27 = tmp12 * tmp1
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp30 = tmp23 + tmp29
    tmp31 = tmp24 + tmp30
    tmp32 = tmp31 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x2 + (784*y3)), tmp31, xmask & ymask)
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y2/cy2mp2zcv4lsecnzbssutdvcvuz6jkdxjdkquwvrmw6ghqthjpdd.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_per_fused_native_group_norm_backward_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 - tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kb/ckbvzebbq22yj36hvbewhtefr3fhshiajyceqiwqpxplcwqnu33c.py
# Source Nodes: [sub_11], Original ATen: [aten.mul, aten.sub, aten.sum]
# sub_11 => sub_34
triton_red_fused_mul_sub_sum_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sub_sum_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x0) + (150528*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iu/ciuvbhxw5mdnnfr47r4b6gc2khzl3rqg46rtwskcrz63f7kaodf5.py
# Source Nodes: [sub_11], Original ATen: [aten.mul, aten.sub, aten.sum]
# sub_11 => sub_34
triton_per_fused_mul_sub_sum_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sub_sum_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gh/cghakr4log3twvojqcn7htdv4kjqiaq3ylossuazdfiab5qvttcv.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]

triton_per_fused_avg_pool2d_backward_native_group_norm_backward_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_avg_pool2d_backward_native_group_norm_backward_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 1536
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 28
    r2 = (rindex // 28)
    x0 = xindex
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + ((28*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(28, 2 + r2))))) + (784*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(28, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + ((28*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(28, 2 + r2))))) + (784*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(28, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr0 + ((28*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(28, 2 + r2))))) + (784*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(28, 2 + r1))))), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr0 + ((28*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(28, 2 + r2))))) + (784*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(28, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr0 + ((28*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(28, 2 + r2))))) + (784*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(28, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr0 + ((28*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(28, 2 + r2))))) + (784*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(28, 2 + r1))))), rmask & xmask, other=0.0)
    tmp48 = tl.load(in_ptr0 + ((28*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(28, 2 + r2))))) + (784*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(28, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr0 + ((28*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(28, 2 + r2))))) + (784*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(28, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.load(in_ptr0 + ((28*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(28, 2 + r2))))) + (784*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(28, 2 + r1))))), rmask & xmask, other=0.0)
    tmp68 = tl.load(in_ptr0 + (r3 + (784*x0)), rmask & xmask, other=0.0)
    tmp71 = tl.load(in_ptr1 + (r3 + (784*x0)), rmask & xmask, other=0.0)
    tmp1 = ((tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(28, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(28, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(28, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(28, 2 + (tl.math.max(0, (-1) + r1)))))
    tmp2 = tmp0 / tmp1
    tmp3 = tl.math.max(0, (-1) + r2)
    tmp4 = tl.math.min(28, 2 + r2)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.math.max(0, (-1) + r1)
    tmp7 = tl.math.min(28, 2 + r1)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp5 & tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp2, tmp10)
    tmp13 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(28, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(28, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r1)))))
    tmp14 = tmp12 / tmp13
    tmp15 = 1 + (tl.math.max(0, (-1) + r1))
    tmp16 = tmp15 < tmp7
    tmp17 = tmp5 & tmp16
    tmp18 = tmp11 + tmp14
    tmp19 = tl.where(tmp17, tmp18, tmp11)
    tmp21 = ((-1)*(tl.math.min(28, 2 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(28, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(28, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))
    tmp22 = tmp20 / tmp21
    tmp23 = 2 + (tl.math.max(0, (-1) + r1))
    tmp24 = tmp23 < tmp7
    tmp25 = tmp5 & tmp24
    tmp26 = tmp19 + tmp22
    tmp27 = tl.where(tmp25, tmp26, tmp19)
    tmp29 = ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(28, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(28, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r2)))))
    tmp30 = tmp28 / tmp29
    tmp31 = 1 + (tl.math.max(0, (-1) + r2))
    tmp32 = tmp31 < tmp4
    tmp33 = tmp32 & tmp8
    tmp34 = tmp27 + tmp30
    tmp35 = tl.where(tmp33, tmp34, tmp27)
    tmp37 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(28, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r1)))))
    tmp38 = tmp36 / tmp37
    tmp39 = tmp32 & tmp16
    tmp40 = tmp35 + tmp38
    tmp41 = tl.where(tmp39, tmp40, tmp35)
    tmp43 = ((-1)*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(28, 3 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r2))
    tmp44 = tmp42 / tmp43
    tmp45 = tmp32 & tmp24
    tmp46 = tmp41 + tmp44
    tmp47 = tl.where(tmp45, tmp46, tmp41)
    tmp49 = ((-1)*(tl.math.min(28, 2 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(28, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(28, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r2))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))
    tmp50 = tmp48 / tmp49
    tmp51 = 2 + (tl.math.max(0, (-1) + r2))
    tmp52 = tmp51 < tmp4
    tmp53 = tmp52 & tmp8
    tmp54 = tmp47 + tmp50
    tmp55 = tl.where(tmp53, tmp54, tmp47)
    tmp57 = ((-1)*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(28, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1))
    tmp58 = tmp56 / tmp57
    tmp59 = tmp52 & tmp16
    tmp60 = tmp55 + tmp58
    tmp61 = tl.where(tmp59, tmp60, tmp55)
    tmp63 = 1 + ((-1)*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(28, 4 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1)) + (tl.math.max(0, (-1) + r2))
    tmp64 = tmp62 / tmp63
    tmp65 = tmp52 & tmp24
    tmp66 = tmp61 + tmp64
    tmp67 = tl.where(tmp65, tmp66, tmp61)
    tmp69 = -tmp68
    tmp70 = tmp69 + tmp67
    tmp72 = tmp70 * tmp71
    tmp73 = tl.broadcast_to(tmp72, [RBLOCK])
    tmp75 = tl.where(rmask & xmask, tmp73, 0)
    tmp76 = triton_helpers.promote_to_tensor(tl.sum(tmp75, 0))
    tmp77 = tl.broadcast_to(tmp70, [RBLOCK])
    tmp79 = tl.where(rmask & xmask, tmp77, 0)
    tmp80 = triton_helpers.promote_to_tensor(tl.sum(tmp79, 0))
    tl.store(out_ptr0 + (r3 + (784*x0)), tmp67, rmask & xmask)
    tl.store(out_ptr1 + (x0), tmp76, xmask)
    tl.store(out_ptr2 + (x0), tmp80, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k6/ck6njvgb537pa7yyri7d6vs7phqe6auqbhjqisbwjwhhs3gvx4py.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_66', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 150528)
    x1 = (xindex // 784) % 192
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x3), None)
    tmp10 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = -tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 * tmp7
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 - tmp13
    tmp15 = tmp14 * tmp5
    tmp16 = tmp15 * tmp5
    tmp17 = tmp16 * tmp5
    tmp18 = 6.64328231292517e-06
    tmp19 = tmp17 * tmp18
    tmp20 = tmp9 * tmp19
    tmp21 = tmp8 + tmp20
    tmp22 = -tmp19
    tmp23 = tmp22 * tmp11
    tmp24 = tmp10 * tmp5
    tmp25 = tmp24 * tmp18
    tmp26 = tmp23 - tmp25
    tmp27 = tmp21 + tmp26
    tmp28 = tmp0 + tmp27
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/af/cafmqmf64mcnfdugxiwyf4nstcgtidq5cyncnqjpptipasaytm2a.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_67', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 150528)
    x1 = (xindex // 784) % 192
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x3), None)
    tmp10 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = -tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 * tmp7
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 - tmp13
    tmp15 = tmp14 * tmp5
    tmp16 = tmp15 * tmp5
    tmp17 = tmp16 * tmp5
    tmp18 = 6.64328231292517e-06
    tmp19 = tmp17 * tmp18
    tmp20 = tmp9 * tmp19
    tmp21 = tmp8 + tmp20
    tmp22 = -tmp19
    tmp23 = tmp22 * tmp11
    tmp24 = tmp10 * tmp5
    tmp25 = tmp24 * tmp18
    tmp26 = tmp23 - tmp25
    tmp27 = tmp21 + tmp26
    tmp28 = tmp0 + tmp27
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/n6/cn6fodbqhhw5v24xndsimrsgqkkoxkwnmzrgcftqevf2adprkxce.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_red_fused_native_group_norm_backward_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_backward_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10752
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192) % 7
    x2 = (xindex // 1344)
    x4 = (xindex // 192)
    tmp5 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (112*x1) + (784*x0) + (150528*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (192*r3) + (21504*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (192*r3) + (21504*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr3 + (x0 + (192*r3) + (21504*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tmp2 - tmp3
        tmp6 = tmp4 * tmp5
        tmp7 = tmp1 + tmp6
        tmp8 = tmp0 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kh/ckh767yuuwfnic5i26bodckgybpmmruy67z7h5nsziwqh72yxe7y.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_per_fused_native_group_norm_backward_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 192
    x1 = (xindex // 192)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (1344*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o5/co5mbegukbfhuipz2bazxig4cauynflx67cvsbhgrfvxvorj7zxh.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_poi_fused_native_group_norm_backward_70 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_backward_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr5 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr7 + (y1), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr8 + (y1), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr9 + (y1), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp8 = tmp6 - tmp7
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 + tmp10
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp1
    tmp18 = tmp17 * tmp1
    tmp19 = tmp18 * tmp1
    tmp20 = 6.64328231292517e-06
    tmp21 = tmp19 * tmp20
    tmp22 = tmp11 * tmp21
    tmp23 = tmp4 + tmp22
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uv/cuvjqnufj23donrxzahq7qer5fw2rc4rcze5dknimbi4hj42flkz.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_71', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y1), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y1), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y1), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 * tmp7
    tmp10 = tmp9 * tmp7
    tmp11 = 6.64328231292517e-06
    tmp12 = tmp10 * tmp11
    tmp13 = -tmp12
    tmp14 = tmp13 * tmp3
    tmp15 = tmp2 * tmp7
    tmp16 = tmp15 * tmp11
    tmp17 = tmp14 - tmp16
    tmp18 = tmp1 + tmp17
    tmp19 = tmp0 + tmp18
    tmp21 = tmp19 * tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp19, xmask & ymask)
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lz/clzk7gocvv76ijmkownnyhkmlo2hmx32vgxagcfs2hi2bihix63e.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]

triton_per_fused_avg_pool2d_backward_native_group_norm_backward_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_avg_pool2d_backward_native_group_norm_backward_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 1536
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 28
    r2 = (rindex // 28)
    x0 = xindex
    r3 = rindex
    x4 = xindex % 192
    x5 = (xindex // 192)
    tmp0 = tl.load(in_ptr0 + ((28*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(28, 2 + r2))))) + (784*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(28, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + ((28*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(28, 2 + r2))))) + (784*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(28, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr0 + ((28*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(28, 2 + r2))))) + (784*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(28, 2 + r1))))), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr0 + ((28*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(28, 2 + r2))))) + (784*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(28, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr0 + ((28*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(28, 2 + r2))))) + (784*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(28, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr0 + ((28*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(28, 2 + r2))))) + (784*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(28, 2 + r1))))), rmask & xmask, other=0.0)
    tmp48 = tl.load(in_ptr0 + ((28*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(28, 2 + r2))))) + (784*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(28, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr0 + ((28*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(28, 2 + r2))))) + (784*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(28, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.load(in_ptr0 + ((28*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(28, 2 + r2))))) + (784*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(28, 2 + r1))))), rmask & xmask, other=0.0)
    tmp68 = tl.load(in_ptr0 + (r3 + (784*x0)), rmask & xmask, other=0.0)
    tmp71 = tl.load(in_ptr1 + (x4 + (192*r3) + (150528*x5)), rmask & xmask, other=0.0)
    tmp1 = ((tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(28, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(28, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(28, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(28, 2 + (tl.math.max(0, (-1) + r1)))))
    tmp2 = tmp0 / tmp1
    tmp3 = tl.math.max(0, (-1) + r2)
    tmp4 = tl.math.min(28, 2 + r2)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.math.max(0, (-1) + r1)
    tmp7 = tl.math.min(28, 2 + r1)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp5 & tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp2, tmp10)
    tmp13 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(28, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(28, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r1)))))
    tmp14 = tmp12 / tmp13
    tmp15 = 1 + (tl.math.max(0, (-1) + r1))
    tmp16 = tmp15 < tmp7
    tmp17 = tmp5 & tmp16
    tmp18 = tmp11 + tmp14
    tmp19 = tl.where(tmp17, tmp18, tmp11)
    tmp21 = ((-1)*(tl.math.min(28, 2 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(28, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(28, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))
    tmp22 = tmp20 / tmp21
    tmp23 = 2 + (tl.math.max(0, (-1) + r1))
    tmp24 = tmp23 < tmp7
    tmp25 = tmp5 & tmp24
    tmp26 = tmp19 + tmp22
    tmp27 = tl.where(tmp25, tmp26, tmp19)
    tmp29 = ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(28, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(28, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r2)))))
    tmp30 = tmp28 / tmp29
    tmp31 = 1 + (tl.math.max(0, (-1) + r2))
    tmp32 = tmp31 < tmp4
    tmp33 = tmp32 & tmp8
    tmp34 = tmp27 + tmp30
    tmp35 = tl.where(tmp33, tmp34, tmp27)
    tmp37 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(28, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r1)))))
    tmp38 = tmp36 / tmp37
    tmp39 = tmp32 & tmp16
    tmp40 = tmp35 + tmp38
    tmp41 = tl.where(tmp39, tmp40, tmp35)
    tmp43 = ((-1)*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(28, 3 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r2))
    tmp44 = tmp42 / tmp43
    tmp45 = tmp32 & tmp24
    tmp46 = tmp41 + tmp44
    tmp47 = tl.where(tmp45, tmp46, tmp41)
    tmp49 = ((-1)*(tl.math.min(28, 2 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(28, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(28, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r2))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))
    tmp50 = tmp48 / tmp49
    tmp51 = 2 + (tl.math.max(0, (-1) + r2))
    tmp52 = tmp51 < tmp4
    tmp53 = tmp52 & tmp8
    tmp54 = tmp47 + tmp50
    tmp55 = tl.where(tmp53, tmp54, tmp47)
    tmp57 = ((-1)*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(28, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(28, 3 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1))
    tmp58 = tmp56 / tmp57
    tmp59 = tmp52 & tmp16
    tmp60 = tmp55 + tmp58
    tmp61 = tl.where(tmp59, tmp60, tmp55)
    tmp63 = 1 + ((-1)*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(28, 4 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(28, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1)) + (tl.math.max(0, (-1) + r2))
    tmp64 = tmp62 / tmp63
    tmp65 = tmp52 & tmp24
    tmp66 = tmp61 + tmp64
    tmp67 = tl.where(tmp65, tmp66, tmp61)
    tmp69 = -tmp68
    tmp70 = tmp69 + tmp67
    tmp72 = tmp70 * tmp71
    tmp73 = tl.broadcast_to(tmp72, [RBLOCK])
    tmp75 = tl.where(rmask & xmask, tmp73, 0)
    tmp76 = triton_helpers.promote_to_tensor(tl.sum(tmp75, 0))
    tmp77 = tl.broadcast_to(tmp70, [RBLOCK])
    tmp79 = tl.where(rmask & xmask, tmp77, 0)
    tmp80 = triton_helpers.promote_to_tensor(tl.sum(tmp79, 0))
    tl.store(out_ptr0 + (r3 + (784*x0)), tmp67, rmask & xmask)
    tl.store(out_ptr1 + (x0), tmp76, xmask)
    tl.store(out_ptr2 + (x0), tmp80, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sj/csjszridhgigx7fv6n4tvmxi47wm7r6obxdsunokhotcrog74ey7.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_73 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_73', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 192)
    y0 = yindex % 192
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y1), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y1), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (y1), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (y1), ymask, eviction_policy='evict_last')
    tmp2 = -tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 * tmp7
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 - tmp13
    tmp15 = tmp14 * tmp5
    tmp16 = tmp15 * tmp5
    tmp17 = tmp16 * tmp5
    tmp18 = 6.64328231292517e-06
    tmp19 = tmp17 * tmp18
    tmp20 = tmp9 * tmp19
    tmp21 = tmp8 + tmp20
    tmp22 = -tmp19
    tmp23 = tmp22 * tmp11
    tmp24 = tmp10 * tmp5
    tmp25 = tmp24 * tmp18
    tmp26 = tmp23 - tmp25
    tmp27 = tmp21 + tmp26
    tmp28 = tmp0 + tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp28, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hs/chsm5kuuiyoq7xmnf7r7ekpyznjscybhst7qdngz33hyi2tvfx7o.py
# Source Nodes: [], Original ATen: [aten.mul]

triton_poi_fused_mul_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 96
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3q/c3qageazeqdtgk5vfjzxjslfuvm3awobk5tot5bi667jtzuzmeol.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18816
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x1) + (301056*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (96*r2) + (12288*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2g/c2g55axe5hfldbstqpnsdzy6ejj2xndokyg5mk2fakyxdqaf6kll.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_per_fused_mul_sum_76 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7p/c7pv6knsletuqbhr4wjt7mbmvedcfizw5moe7yex6fug3cyidglq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 96
    x1 = (xindex // 96)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (301056*(r2 // 3136)) + (602112*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/de/cdeay3xrlzcnbvrywipufoenfiiotmhfcgqiuzvvh7yanomzgnph.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_78 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ln/clnninp2ov472mvfu7nyawucqisvr5estuvqrlil3sztdapsqudz.py
# Source Nodes: [x_46], Original ATen: [aten.gelu, aten.gelu_backward]
# x_46 => add_40, erf_5, mul_51
triton_poi_fused_gelu_gelu_backward_79 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_79', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (384*x2) + (1204224*y1)), xmask, eviction_policy='evict_last')
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (3136*y3)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ug/cugjnvaywkaofqu3x3fwmthuxrergc5a2pkvid53cqwmo7e5uii3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_80 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (1204224*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4n/c4nfsgapv5onhgvlhw2i5csev4oqxwkyyx7xaexbblmyjeepdln4.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_red_fused_native_group_norm_backward_81 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_backward_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19200
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 25
    x4 = (xindex // 25)
    x1 = (xindex // 25) % 96
    x2 = (xindex // 2400)
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x0)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((3136*x4) + ((r3 + (126*x0)) % 3136)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((3136*x4) + ((r3 + (126*x0)) % 3136)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x1 + (96*((r3 + (126*x0)) % 3136)) + (301056*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x1 + (96*((r3 + (126*x0)) % 3136)) + (301056*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp5 - tmp6
        tmp8 = tl.load(in_ptr4 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 * tmp8
        tmp10 = tmp4 + tmp9
        tmp11 = tmp3 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33taw7bko3mjpazl552dlzfa3tuosds7s7qh5qzwczu25zazivy.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_per_fused_native_group_norm_backward_82 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_82', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (25*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gg/cggy3qpvfjjimeq7rrxmsqvwc22duhenfmbiz3ohrf7hfaqkcgy2.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_red_fused_native_group_norm_backward_83 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_backward_83', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mr/cmrbyvk5f4uu4biorvia4upeqm7d3xgz7a6o7qw55fhfl3j467w2.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_per_fused_native_group_norm_backward_84 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_84', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 96
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (96*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (96*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp7 * tmp1
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/74/c74bxz3egdk2zg5qndxig7qmfckkhskaf5qdzgtaxvbzbpkb7zka.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]

triton_poi_fused_add_mul_native_group_norm_backward_85 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_group_norm_backward_85', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 96)
    y0 = yindex % 96
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y1), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0 + (96*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0 + (96*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (y1), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (y1), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr8 + (y1), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_out_ptr1 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp8 = tmp6 - tmp7
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 + tmp10
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp1
    tmp18 = tmp17 * tmp1
    tmp19 = tmp18 * tmp1
    tmp20 = 3.321641156462585e-06
    tmp21 = tmp19 * tmp20
    tmp22 = tmp11 * tmp21
    tmp23 = tmp4 + tmp22
    tmp25 = -tmp21
    tmp26 = tmp25 * tmp13
    tmp27 = tmp12 * tmp1
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp30 = tmp23 + tmp29
    tmp31 = tmp24 + tmp30
    tmp32 = tmp31 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x2 + (3136*y3)), tmp31, xmask & ymask)
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w6/cw6lkotofnxxp5aaxrvrcutiabyuhyezi4fkxjlfx7qiw6nkxeyz.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_per_fused_native_group_norm_backward_86 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_86', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (96*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 - tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x2/cx2xoqdnaw3gbqyq2gxrunptpkjuzh6rj5l4c2mfsb3zjkdabnrm.py
# Source Nodes: [sub_5], Original ATen: [aten.mul, aten.sub, aten.sum]
# sub_5 => sub_16
triton_red_fused_mul_sub_sum_87 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sub_sum_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18816
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 96
    x1 = (xindex // 96)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (301056*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cx/ccxh7bgpeoc3c36mg666bjblqlfalq7qjxl7rx47zenetlssgf4l.py
# Source Nodes: [sub_5], Original ATen: [aten.mul, aten.sub, aten.sum]
# sub_5 => sub_16
triton_red_fused_mul_sub_sum_88 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sub_sum_88', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/77/c77tpn3eml2tiwurpds4yh3x63btp6djdygx6eaqtcn5ao6ty725.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]

triton_red_fused_avg_pool2d_backward_native_group_norm_backward_89 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_avg_pool2d_backward_native_group_norm_backward_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp74 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 56
        r2 = (rindex // 56)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(56, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(56, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(56, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(56, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp36 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(56, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp42 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(56, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp48 = tl.load(in_ptr0 + ((56*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(56, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp56 = tl.load(in_ptr0 + ((56*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(56, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp62 = tl.load(in_ptr0 + ((56*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(56, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp68 = tl.load(in_ptr0 + (r3 + (3136*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp71 = tl.load(in_ptr1 + (r3 + (3136*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = ((tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1)))))
        tmp2 = tmp0 / tmp1
        tmp3 = tl.math.max(0, (-1) + r2)
        tmp4 = tl.math.min(56, 2 + r2)
        tmp5 = tmp3 < tmp4
        tmp6 = tl.math.max(0, (-1) + r1)
        tmp7 = tl.math.min(56, 2 + r1)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp5 & tmp8
        tmp10 = 0.0
        tmp11 = tl.where(tmp9, tmp2, tmp10)
        tmp13 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1)))))
        tmp14 = tmp12 / tmp13
        tmp15 = 1 + (tl.math.max(0, (-1) + r1))
        tmp16 = tmp15 < tmp7
        tmp17 = tmp5 & tmp16
        tmp18 = tmp11 + tmp14
        tmp19 = tl.where(tmp17, tmp18, tmp11)
        tmp21 = ((-1)*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))
        tmp22 = tmp20 / tmp21
        tmp23 = 2 + (tl.math.max(0, (-1) + r1))
        tmp24 = tmp23 < tmp7
        tmp25 = tmp5 & tmp24
        tmp26 = tmp19 + tmp22
        tmp27 = tl.where(tmp25, tmp26, tmp19)
        tmp29 = ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2)))))
        tmp30 = tmp28 / tmp29
        tmp31 = 1 + (tl.math.max(0, (-1) + r2))
        tmp32 = tmp31 < tmp4
        tmp33 = tmp32 & tmp8
        tmp34 = tmp27 + tmp30
        tmp35 = tl.where(tmp33, tmp34, tmp27)
        tmp37 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1)))))
        tmp38 = tmp36 / tmp37
        tmp39 = tmp32 & tmp16
        tmp40 = tmp35 + tmp38
        tmp41 = tl.where(tmp39, tmp40, tmp35)
        tmp43 = ((-1)*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r2))
        tmp44 = tmp42 / tmp43
        tmp45 = tmp32 & tmp24
        tmp46 = tmp41 + tmp44
        tmp47 = tl.where(tmp45, tmp46, tmp41)
        tmp49 = ((-1)*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))
        tmp50 = tmp48 / tmp49
        tmp51 = 2 + (tl.math.max(0, (-1) + r2))
        tmp52 = tmp51 < tmp4
        tmp53 = tmp52 & tmp8
        tmp54 = tmp47 + tmp50
        tmp55 = tl.where(tmp53, tmp54, tmp47)
        tmp57 = ((-1)*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1))
        tmp58 = tmp56 / tmp57
        tmp59 = tmp52 & tmp16
        tmp60 = tmp55 + tmp58
        tmp61 = tl.where(tmp59, tmp60, tmp55)
        tmp63 = 1 + ((-1)*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1)) + (tl.math.max(0, (-1) + r2))
        tmp64 = tmp62 / tmp63
        tmp65 = tmp52 & tmp24
        tmp66 = tmp61 + tmp64
        tmp67 = tl.where(tmp65, tmp66, tmp61)
        tmp69 = -tmp68
        tmp70 = tmp69 + tmp67
        tmp72 = tmp70 * tmp71
        tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
        tmp75 = _tmp74 + tmp73
        _tmp74 = tl.where(rmask & xmask, tmp75, _tmp74)
        tl.store(out_ptr0 + (r3 + (3136*x0)), tmp67, rmask & xmask)
    tmp74 = tl.sum(_tmp74, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp74, xmask)
    _tmp81 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp76 = tl.load(in_ptr0 + (r3 + (3136*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp78 = tl.load(out_ptr0 + (r3 + (3136*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp77 = -tmp76
        tmp79 = tmp77 + tmp78
        tmp80 = tl.broadcast_to(tmp79, [XBLOCK, RBLOCK])
        tmp82 = _tmp81 + tmp80
        _tmp81 = tl.where(rmask & xmask, tmp82, _tmp81)
    tmp81 = tl.sum(_tmp81, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp81, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zw/czwmm53brqcxnzpkdyqdhke2hhv2jcxqrxld2ii7bvntqlbnepiz.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_90 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_90', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 301056)
    x1 = (xindex // 3136) % 96
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x3), None)
    tmp10 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = -tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 * tmp7
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 - tmp13
    tmp15 = tmp14 * tmp5
    tmp16 = tmp15 * tmp5
    tmp17 = tmp16 * tmp5
    tmp18 = 3.321641156462585e-06
    tmp19 = tmp17 * tmp18
    tmp20 = tmp9 * tmp19
    tmp21 = tmp8 + tmp20
    tmp22 = -tmp19
    tmp23 = tmp22 * tmp11
    tmp24 = tmp10 * tmp5
    tmp25 = tmp24 * tmp18
    tmp26 = tmp23 - tmp25
    tmp27 = tmp21 + tmp26
    tmp28 = tmp0 + tmp27
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m4/cm4dej6h27hlv53pdkfgzalww3envmg3ibodnc2ndfyvf6ac7vsa.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_91 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_91', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 301056)
    x1 = (xindex // 3136) % 96
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x3), None)
    tmp10 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = -tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 * tmp7
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 - tmp13
    tmp15 = tmp14 * tmp5
    tmp16 = tmp15 * tmp5
    tmp17 = tmp16 * tmp5
    tmp18 = 3.321641156462585e-06
    tmp19 = tmp17 * tmp18
    tmp20 = tmp9 * tmp19
    tmp21 = tmp8 + tmp20
    tmp22 = -tmp19
    tmp23 = tmp22 * tmp11
    tmp24 = tmp10 * tmp5
    tmp25 = tmp24 * tmp18
    tmp26 = tmp23 - tmp25
    tmp27 = tmp21 + tmp26
    tmp28 = tmp0 + tmp27
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/n5/cn5ea4735xzw32tubqzpyxalifmilzupsgiwlxgbybgquay6vw5r.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_red_fused_native_group_norm_backward_92 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_backward_92', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19200
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96) % 25
    x0 = xindex % 96
    x2 = (xindex // 2400)
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((3136*x0) + (301056*x2) + ((r3 + (126*x1)) % 3136)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (96*((r3 + (126*x1)) % 3136)) + (301056*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (96*((r3 + (126*x1)) % 3136)) + (301056*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x0 + (96*((r3 + (126*x1)) % 3136)) + (301056*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp5 - tmp6
        tmp8 = tl.load(in_ptr4 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 * tmp8
        tmp10 = tmp4 + tmp9
        tmp11 = tmp3 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f6/cf6g6qdvqtqc7bycl6e2djrd37tl74ot5jgilcq7agnmpqg6reah.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_per_fused_native_group_norm_backward_93 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 32],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_backward_93', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 96
    x1 = (xindex // 96)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (96*r2) + (2400*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d3/cd3typptzcukwvpkiax4p25ol5etqolvokzkpoqo4ys6w6q3mf4c.py
# Source Nodes: [], Original ATen: [aten.native_group_norm_backward]

triton_poi_fused_native_group_norm_backward_94 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_backward_94', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 96
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr5 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr7 + (y1), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr8 + (y1), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr9 + (y1), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp8 = tmp6 - tmp7
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 + tmp10
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp1
    tmp18 = tmp17 * tmp1
    tmp19 = tmp18 * tmp1
    tmp20 = 3.321641156462585e-06
    tmp21 = tmp19 * tmp20
    tmp22 = tmp11 * tmp21
    tmp23 = tmp4 + tmp22
    tl.store(out_ptr0 + (x2 + (96*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7m/c7mmmmqp6bspqu4miezi32427q3vm5pl4rxh7e7egypwogumzeoq.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_95 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_95', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (96*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y1), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y1), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y1), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 * tmp7
    tmp10 = tmp9 * tmp7
    tmp11 = 3.321641156462585e-06
    tmp12 = tmp10 * tmp11
    tmp13 = -tmp12
    tmp14 = tmp13 * tmp3
    tmp15 = tmp2 * tmp7
    tmp16 = tmp15 * tmp11
    tmp17 = tmp14 - tmp16
    tmp18 = tmp1 + tmp17
    tmp19 = tmp0 + tmp18
    tmp21 = tmp19 * tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (3136*y3)), tmp19, xmask & ymask)
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/of/coflepsbflvjhck3uza6fodmys7h633u7febwk5i24eti3t6nuvq.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]

triton_red_fused_avg_pool2d_backward_native_group_norm_backward_96 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_avg_pool2d_backward_native_group_norm_backward_96', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    x4 = xindex % 96
    x5 = (xindex // 96)
    _tmp74 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 56
        r2 = (rindex // 56)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(56, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(56, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (-1) + r2), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(56, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(56, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp36 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(56, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp42 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(56, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp48 = tl.load(in_ptr0 + ((56*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(tl.math.max(0, (-1) + r1), (-1) + (tl.math.min(56, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp56 = tl.load(in_ptr0 + ((56*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(1 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(56, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp62 = tl.load(in_ptr0 + ((56*(tl.math.min(2 + (tl.math.max(0, (-1) + r2)), (-1) + (tl.math.min(56, 2 + r2))))) + (3136*x0) + (tl.math.min(2 + (tl.math.max(0, (-1) + r1)), (-1) + (tl.math.min(56, 2 + r1))))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp68 = tl.load(in_ptr0 + (r3 + (3136*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp71 = tl.load(in_ptr1 + (x4 + (96*r3) + (301056*x5)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = ((tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1)))))
        tmp2 = tmp0 / tmp1
        tmp3 = tl.math.max(0, (-1) + r2)
        tmp4 = tl.math.min(56, 2 + r2)
        tmp5 = tmp3 < tmp4
        tmp6 = tl.math.max(0, (-1) + r1)
        tmp7 = tl.math.min(56, 2 + r1)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp5 & tmp8
        tmp10 = 0.0
        tmp11 = tl.where(tmp9, tmp2, tmp10)
        tmp13 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1)))))
        tmp14 = tmp12 / tmp13
        tmp15 = 1 + (tl.math.max(0, (-1) + r1))
        tmp16 = tmp15 < tmp7
        tmp17 = tmp5 & tmp16
        tmp18 = tmp11 + tmp14
        tmp19 = tl.where(tmp17, tmp18, tmp11)
        tmp21 = ((-1)*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))) + ((tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r2))))
        tmp22 = tmp20 / tmp21
        tmp23 = 2 + (tl.math.max(0, (-1) + r1))
        tmp24 = tmp23 < tmp7
        tmp25 = tmp5 & tmp24
        tmp26 = tmp19 + tmp22
        tmp27 = tl.where(tmp25, tmp26, tmp19)
        tmp29 = ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2)))))
        tmp30 = tmp28 / tmp29
        tmp31 = 1 + (tl.math.max(0, (-1) + r2))
        tmp32 = tmp31 < tmp4
        tmp33 = tmp32 & tmp8
        tmp34 = tmp27 + tmp30
        tmp35 = tl.where(tmp33, tmp34, tmp27)
        tmp37 = ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1)))))
        tmp38 = tmp36 / tmp37
        tmp39 = tmp32 & tmp16
        tmp40 = tmp35 + tmp38
        tmp41 = tl.where(tmp39, tmp40, tmp35)
        tmp43 = ((-1)*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r2))
        tmp44 = tmp42 / tmp43
        tmp45 = tmp32 & tmp24
        tmp46 = tmp41 + tmp44
        tmp47 = tl.where(tmp45, tmp46, tmp41)
        tmp49 = ((-1)*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r2))*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))) + ((tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(56, 2 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + (tl.math.max(0, (-1) + (tl.math.max(0, (-1) + r1))))
        tmp50 = tmp48 / tmp49
        tmp51 = 2 + (tl.math.max(0, (-1) + r2))
        tmp52 = tmp51 < tmp4
        tmp53 = tmp52 & tmp8
        tmp54 = tmp47 + tmp50
        tmp55 = tl.where(tmp53, tmp54, tmp47)
        tmp57 = ((-1)*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(56, 3 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1))
        tmp58 = tmp56 / tmp57
        tmp59 = tmp52 & tmp16
        tmp60 = tmp55 + tmp58
        tmp61 = tl.where(tmp59, tmp60, tmp55)
        tmp63 = 1 + ((-1)*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))) + ((-1)*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + ((tl.math.max(0, (-1) + r1))*(tl.math.max(0, (-1) + r2))) + ((tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r1))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r2))))) + ((-1)*(tl.math.max(0, (-1) + r2))*(tl.math.min(56, 4 + (tl.math.max(0, (-1) + r1))))) + (tl.math.max(0, (-1) + r1)) + (tl.math.max(0, (-1) + r2))
        tmp64 = tmp62 / tmp63
        tmp65 = tmp52 & tmp24
        tmp66 = tmp61 + tmp64
        tmp67 = tl.where(tmp65, tmp66, tmp61)
        tmp69 = -tmp68
        tmp70 = tmp69 + tmp67
        tmp72 = tmp70 * tmp71
        tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
        tmp75 = _tmp74 + tmp73
        _tmp74 = tl.where(rmask & xmask, tmp75, _tmp74)
        tl.store(out_ptr0 + (r3 + (3136*x0)), tmp67, rmask & xmask)
    tmp74 = tl.sum(_tmp74, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp74, xmask)
    _tmp81 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp76 = tl.load(in_ptr0 + (r3 + (3136*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp78 = tl.load(out_ptr0 + (r3 + (3136*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp77 = -tmp76
        tmp79 = tmp77 + tmp78
        tmp80 = tl.broadcast_to(tmp79, [XBLOCK, RBLOCK])
        tmp82 = _tmp81 + tmp80
        _tmp81 = tl.where(rmask & xmask, tmp82, _tmp81)
    tmp81 = tl.sum(_tmp81, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp81, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7n/c7nvsnto4epmn4fa27f37jrdjbk7x3ldpaifwieg4fujb5pr6mfw.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_97 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_97', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 96)
    y0 = yindex % 96
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y1), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0 + (96*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y1), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (y1), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (y1), ymask, eviction_policy='evict_last')
    tmp2 = -tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 * tmp7
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 - tmp13
    tmp15 = tmp14 * tmp5
    tmp16 = tmp15 * tmp5
    tmp17 = tmp16 * tmp5
    tmp18 = 3.321641156462585e-06
    tmp19 = tmp17 * tmp18
    tmp20 = tmp9 * tmp19
    tmp21 = tmp8 + tmp20
    tmp22 = -tmp19
    tmp23 = tmp22 * tmp11
    tmp24 = tmp10 * tmp5
    tmp25 = tmp24 * tmp18
    tmp26 = tmp23 - tmp25
    tmp27 = tmp21 + tmp26
    tmp28 = tmp0 + tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (3136*y3)), tmp28, xmask & ymask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_141, primals_142, primals_144, primals_145, primals_147, primals_148, primals_150, primals_151, primals_153, primals_154, primals_156, primals_157, primals_159, primals_160, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_221, primals_223, primals_225, primals_227, primals_229, primals_231, primals_233, primals_235, primals_237, primals_239, primals_241, primals_243, primals_245, primals_247, primals_249, primals_251, primals_253, primals_255, primals_257, primals_259, primals_261, primals_263, primals_265, primals_267, primals_269, primals_271, primals_273, primals_275, primals_277, primals_279, primals_281, primals_283, primals_285, primals_287, primals_289, primals_291, primals_293, primals_295, primals_297, primals_299, primals_301, primals_303, primals_305, primals_307, primals_309, primals_311, primals_313, primals_315, primals_317, primals_319, primals_321, primals_323, primals_325, primals_327, primals_329, primals_331, primals_333, primals_335, primals_337, primals_339, primals_341, primals_343, primals_345, primals_347, primals_349, primals_351, primals_353, primals_355, primals_357, primals_359, primals_361, primals_363, primals_365, primals_367, primals_369, primals_373, convolution, add_1, avg_pool2d, add_4, convolution_1, clone, convolution_2, add_8, avg_pool2d_1, add_11, convolution_3, clone_2, convolution_4, add_15, avg_pool2d_2, add_18, convolution_5, clone_4, convolution_6, add_22, avg_pool2d_3, add_25, convolution_7, clone_6, convolution_8, add_29, avg_pool2d_4, add_32, convolution_9, clone_8, convolution_10, add_36, avg_pool2d_5, add_39, convolution_11, clone_10, convolution_12, add_41, convolution_13, add_43, avg_pool2d_6, add_46, convolution_14, clone_12, convolution_15, add_50, avg_pool2d_7, add_53, convolution_16, clone_14, convolution_17, add_57, avg_pool2d_8, add_60, convolution_18, clone_16, convolution_19, add_64, avg_pool2d_9, add_67, convolution_20, clone_18, convolution_21, add_71, avg_pool2d_10, add_74, convolution_22, clone_20, convolution_23, add_78, avg_pool2d_11, add_81, convolution_24, clone_22, convolution_25, add_83, convolution_26, add_85, avg_pool2d_12, add_88, convolution_27, clone_24, convolution_28, add_92, avg_pool2d_13, add_95, convolution_29, clone_26, convolution_30, add_99, avg_pool2d_14, add_102, convolution_31, clone_28, convolution_32, add_106, avg_pool2d_15, add_109, convolution_33, clone_30, convolution_34, add_113, avg_pool2d_16, add_116, convolution_35, clone_32, convolution_36, add_120, avg_pool2d_17, add_123, convolution_37, clone_34, convolution_38, add_127, avg_pool2d_18, add_130, convolution_39, clone_36, convolution_40, add_134, avg_pool2d_19, add_137, convolution_41, clone_38, convolution_42, add_141, avg_pool2d_20, add_144, convolution_43, clone_40, convolution_44, add_148, avg_pool2d_21, add_151, convolution_45, clone_42, convolution_46, add_155, avg_pool2d_22, add_158, convolution_47, clone_44, convolution_48, add_162, avg_pool2d_23, add_165, convolution_49, clone_46, convolution_50, add_169, avg_pool2d_24, add_172, convolution_51, clone_48, convolution_52, add_176, avg_pool2d_25, add_179, convolution_53, clone_50, convolution_54, add_183, avg_pool2d_26, add_186, convolution_55, clone_52, convolution_56, add_190, avg_pool2d_27, add_193, convolution_57, clone_54, convolution_58, add_197, avg_pool2d_28, add_200, convolution_59, clone_56, convolution_60, add_204, avg_pool2d_29, add_207, convolution_61, clone_58, convolution_62, add_209, convolution_63, add_211, avg_pool2d_30, add_214, convolution_64, clone_60, convolution_65, add_218, avg_pool2d_31, add_221, convolution_66, clone_62, convolution_67, add_225, avg_pool2d_32, add_228, convolution_68, clone_64, convolution_69, add_232, avg_pool2d_33, add_235, convolution_70, clone_66, convolution_71, add_239, avg_pool2d_34, add_242, convolution_72, clone_68, convolution_73, add_246, avg_pool2d_35, add_249, convolution_74, clone_70, convolution_75, mul_324, view_216, permute_3, div, alias_144, alias_145, alias_146, alias_147, alias_148, alias_149, alias_150, alias_151, alias_152, alias_153, alias_154, alias_155, alias_156, alias_157, alias_158, alias_159, alias_160, alias_161, alias_162, alias_163, alias_164, alias_165, alias_166, alias_167, alias_168, alias_169, alias_170, alias_171, alias_172, alias_173, alias_174, alias_175, alias_176, alias_177, alias_178, alias_179, alias_180, alias_181, alias_182, alias_183, alias_184, alias_185, alias_186, alias_187, alias_188, alias_189, alias_190, alias_191, alias_192, alias_193, alias_194, alias_195, alias_196, alias_197, alias_198, alias_199, alias_200, alias_201, alias_202, alias_203, alias_204, alias_205, alias_206, alias_207, alias_208, alias_209, alias_210, alias_211, alias_212, alias_213, alias_214, alias_215, alias_216, alias_217, alias_218, alias_219, alias_220, alias_221, alias_222, alias_223, alias_224, alias_225, alias_226, alias_227, alias_228, alias_229, alias_230, alias_231, alias_232, alias_233, alias_234, alias_235, alias_236, alias_237, alias_238, alias_239, alias_240, alias_241, alias_242, alias_243, alias_244, alias_245, alias_246, alias_247, alias_248, alias_249, alias_250, alias_251, alias_252, alias_253, alias_254, alias_255, alias_256, alias_257, alias_258, alias_259, alias_260, alias_261, alias_262, alias_263, alias_264, alias_265, alias_266, alias_267, alias_268, alias_269, alias_270, alias_271, alias_272, alias_273, alias_274, alias_275, alias_276, alias_277, alias_278, alias_279, alias_280, alias_281, alias_282, alias_283, alias_284, alias_285, alias_286, alias_287, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (96, ), (1, ))
    assert_size_stride(primals_3, (96, ), (1, ))
    assert_size_stride(primals_4, (96, ), (1, ))
    assert_size_stride(primals_6, (96, ), (1, ))
    assert_size_stride(primals_7, (96, ), (1, ))
    assert_size_stride(primals_9, (96, ), (1, ))
    assert_size_stride(primals_10, (96, ), (1, ))
    assert_size_stride(primals_12, (96, ), (1, ))
    assert_size_stride(primals_13, (96, ), (1, ))
    assert_size_stride(primals_15, (96, ), (1, ))
    assert_size_stride(primals_16, (96, ), (1, ))
    assert_size_stride(primals_18, (96, ), (1, ))
    assert_size_stride(primals_19, (96, ), (1, ))
    assert_size_stride(primals_21, (96, ), (1, ))
    assert_size_stride(primals_22, (96, ), (1, ))
    assert_size_stride(primals_24, (96, ), (1, ))
    assert_size_stride(primals_25, (96, ), (1, ))
    assert_size_stride(primals_27, (96, ), (1, ))
    assert_size_stride(primals_28, (96, ), (1, ))
    assert_size_stride(primals_30, (96, ), (1, ))
    assert_size_stride(primals_31, (96, ), (1, ))
    assert_size_stride(primals_33, (96, ), (1, ))
    assert_size_stride(primals_34, (96, ), (1, ))
    assert_size_stride(primals_36, (96, ), (1, ))
    assert_size_stride(primals_37, (192, ), (1, ))
    assert_size_stride(primals_39, (192, ), (1, ))
    assert_size_stride(primals_40, (192, ), (1, ))
    assert_size_stride(primals_42, (192, ), (1, ))
    assert_size_stride(primals_43, (192, ), (1, ))
    assert_size_stride(primals_45, (192, ), (1, ))
    assert_size_stride(primals_46, (192, ), (1, ))
    assert_size_stride(primals_48, (192, ), (1, ))
    assert_size_stride(primals_49, (192, ), (1, ))
    assert_size_stride(primals_51, (192, ), (1, ))
    assert_size_stride(primals_52, (192, ), (1, ))
    assert_size_stride(primals_54, (192, ), (1, ))
    assert_size_stride(primals_55, (192, ), (1, ))
    assert_size_stride(primals_57, (192, ), (1, ))
    assert_size_stride(primals_58, (192, ), (1, ))
    assert_size_stride(primals_60, (192, ), (1, ))
    assert_size_stride(primals_61, (192, ), (1, ))
    assert_size_stride(primals_63, (192, ), (1, ))
    assert_size_stride(primals_64, (192, ), (1, ))
    assert_size_stride(primals_66, (192, ), (1, ))
    assert_size_stride(primals_67, (192, ), (1, ))
    assert_size_stride(primals_69, (192, ), (1, ))
    assert_size_stride(primals_70, (192, ), (1, ))
    assert_size_stride(primals_72, (192, ), (1, ))
    assert_size_stride(primals_73, (384, ), (1, ))
    assert_size_stride(primals_75, (384, ), (1, ))
    assert_size_stride(primals_76, (384, ), (1, ))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_79, (384, ), (1, ))
    assert_size_stride(primals_81, (384, ), (1, ))
    assert_size_stride(primals_82, (384, ), (1, ))
    assert_size_stride(primals_84, (384, ), (1, ))
    assert_size_stride(primals_85, (384, ), (1, ))
    assert_size_stride(primals_87, (384, ), (1, ))
    assert_size_stride(primals_88, (384, ), (1, ))
    assert_size_stride(primals_90, (384, ), (1, ))
    assert_size_stride(primals_91, (384, ), (1, ))
    assert_size_stride(primals_93, (384, ), (1, ))
    assert_size_stride(primals_94, (384, ), (1, ))
    assert_size_stride(primals_96, (384, ), (1, ))
    assert_size_stride(primals_97, (384, ), (1, ))
    assert_size_stride(primals_99, (384, ), (1, ))
    assert_size_stride(primals_100, (384, ), (1, ))
    assert_size_stride(primals_102, (384, ), (1, ))
    assert_size_stride(primals_103, (384, ), (1, ))
    assert_size_stride(primals_105, (384, ), (1, ))
    assert_size_stride(primals_106, (384, ), (1, ))
    assert_size_stride(primals_108, (384, ), (1, ))
    assert_size_stride(primals_109, (384, ), (1, ))
    assert_size_stride(primals_111, (384, ), (1, ))
    assert_size_stride(primals_112, (384, ), (1, ))
    assert_size_stride(primals_114, (384, ), (1, ))
    assert_size_stride(primals_115, (384, ), (1, ))
    assert_size_stride(primals_117, (384, ), (1, ))
    assert_size_stride(primals_118, (384, ), (1, ))
    assert_size_stride(primals_120, (384, ), (1, ))
    assert_size_stride(primals_121, (384, ), (1, ))
    assert_size_stride(primals_123, (384, ), (1, ))
    assert_size_stride(primals_124, (384, ), (1, ))
    assert_size_stride(primals_126, (384, ), (1, ))
    assert_size_stride(primals_127, (384, ), (1, ))
    assert_size_stride(primals_129, (384, ), (1, ))
    assert_size_stride(primals_130, (384, ), (1, ))
    assert_size_stride(primals_132, (384, ), (1, ))
    assert_size_stride(primals_133, (384, ), (1, ))
    assert_size_stride(primals_135, (384, ), (1, ))
    assert_size_stride(primals_136, (384, ), (1, ))
    assert_size_stride(primals_138, (384, ), (1, ))
    assert_size_stride(primals_139, (384, ), (1, ))
    assert_size_stride(primals_141, (384, ), (1, ))
    assert_size_stride(primals_142, (384, ), (1, ))
    assert_size_stride(primals_144, (384, ), (1, ))
    assert_size_stride(primals_145, (384, ), (1, ))
    assert_size_stride(primals_147, (384, ), (1, ))
    assert_size_stride(primals_148, (384, ), (1, ))
    assert_size_stride(primals_150, (384, ), (1, ))
    assert_size_stride(primals_151, (384, ), (1, ))
    assert_size_stride(primals_153, (384, ), (1, ))
    assert_size_stride(primals_154, (384, ), (1, ))
    assert_size_stride(primals_156, (384, ), (1, ))
    assert_size_stride(primals_157, (384, ), (1, ))
    assert_size_stride(primals_159, (384, ), (1, ))
    assert_size_stride(primals_160, (384, ), (1, ))
    assert_size_stride(primals_162, (384, ), (1, ))
    assert_size_stride(primals_163, (384, ), (1, ))
    assert_size_stride(primals_165, (384, ), (1, ))
    assert_size_stride(primals_166, (384, ), (1, ))
    assert_size_stride(primals_168, (384, ), (1, ))
    assert_size_stride(primals_169, (384, ), (1, ))
    assert_size_stride(primals_171, (384, ), (1, ))
    assert_size_stride(primals_172, (384, ), (1, ))
    assert_size_stride(primals_174, (384, ), (1, ))
    assert_size_stride(primals_175, (384, ), (1, ))
    assert_size_stride(primals_177, (384, ), (1, ))
    assert_size_stride(primals_178, (384, ), (1, ))
    assert_size_stride(primals_180, (384, ), (1, ))
    assert_size_stride(primals_181, (768, ), (1, ))
    assert_size_stride(primals_183, (768, ), (1, ))
    assert_size_stride(primals_184, (768, ), (1, ))
    assert_size_stride(primals_186, (768, ), (1, ))
    assert_size_stride(primals_187, (768, ), (1, ))
    assert_size_stride(primals_189, (768, ), (1, ))
    assert_size_stride(primals_190, (768, ), (1, ))
    assert_size_stride(primals_192, (768, ), (1, ))
    assert_size_stride(primals_193, (768, ), (1, ))
    assert_size_stride(primals_195, (768, ), (1, ))
    assert_size_stride(primals_196, (768, ), (1, ))
    assert_size_stride(primals_198, (768, ), (1, ))
    assert_size_stride(primals_199, (768, ), (1, ))
    assert_size_stride(primals_201, (768, ), (1, ))
    assert_size_stride(primals_202, (768, ), (1, ))
    assert_size_stride(primals_204, (768, ), (1, ))
    assert_size_stride(primals_205, (768, ), (1, ))
    assert_size_stride(primals_207, (768, ), (1, ))
    assert_size_stride(primals_208, (768, ), (1, ))
    assert_size_stride(primals_210, (768, ), (1, ))
    assert_size_stride(primals_211, (768, ), (1, ))
    assert_size_stride(primals_213, (768, ), (1, ))
    assert_size_stride(primals_214, (768, ), (1, ))
    assert_size_stride(primals_216, (768, ), (1, ))
    assert_size_stride(primals_217, (768, ), (1, ))
    assert_size_stride(primals_219, (96, 3, 7, 7), (147, 1, 21, 3))
    assert_size_stride(primals_221, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_223, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_225, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_227, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_229, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_231, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_233, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_235, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_237, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_239, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_241, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_243, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_245, (192, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(primals_247, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_249, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_251, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_253, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_255, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_257, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_259, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_261, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_263, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_265, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_267, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_269, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_271, (384, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_273, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_275, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_277, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_279, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_281, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_283, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_285, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_287, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_289, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_291, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_293, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_295, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_297, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_299, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_301, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_303, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_305, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_307, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_309, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_311, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_313, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_315, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_317, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_319, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_321, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_323, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_325, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_327, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_329, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_331, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_333, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_335, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_337, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_339, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_341, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_343, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_345, (768, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_347, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_349, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_351, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_353, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_355, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_357, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_359, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_361, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_363, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_365, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_367, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_369, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_373, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(add_1, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(avg_pool2d, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(add_4, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(convolution_1, (8, 384, 56, 56), (1204224, 1, 21504, 384))
    assert_size_stride(clone, (8, 384, 56, 56), (1204224, 1, 21504, 384))
    assert_size_stride(convolution_2, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(add_8, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(avg_pool2d_1, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(add_11, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(convolution_3, (8, 384, 56, 56), (1204224, 1, 21504, 384))
    assert_size_stride(clone_2, (8, 384, 56, 56), (1204224, 1, 21504, 384))
    assert_size_stride(convolution_4, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(add_15, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(avg_pool2d_2, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(add_18, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(convolution_5, (8, 384, 56, 56), (1204224, 1, 21504, 384))
    assert_size_stride(clone_4, (8, 384, 56, 56), (1204224, 1, 21504, 384))
    assert_size_stride(convolution_6, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(add_22, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(avg_pool2d_3, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(add_25, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(convolution_7, (8, 384, 56, 56), (1204224, 1, 21504, 384))
    assert_size_stride(clone_6, (8, 384, 56, 56), (1204224, 1, 21504, 384))
    assert_size_stride(convolution_8, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(add_29, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(avg_pool2d_4, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(add_32, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(convolution_9, (8, 384, 56, 56), (1204224, 1, 21504, 384))
    assert_size_stride(clone_8, (8, 384, 56, 56), (1204224, 1, 21504, 384))
    assert_size_stride(convolution_10, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(add_36, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(avg_pool2d_5, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(add_39, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(convolution_11, (8, 384, 56, 56), (1204224, 1, 21504, 384))
    assert_size_stride(clone_10, (8, 384, 56, 56), (1204224, 1, 21504, 384))
    assert_size_stride(convolution_12, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(add_41, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(convolution_13, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(add_43, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(avg_pool2d_6, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(add_46, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_14, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(clone_12, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_15, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(add_50, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(avg_pool2d_7, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(add_53, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_16, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(clone_14, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_17, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(add_57, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(avg_pool2d_8, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(add_60, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_18, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(clone_16, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_19, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(add_64, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(avg_pool2d_9, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(add_67, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_20, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(clone_18, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_21, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(add_71, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(avg_pool2d_10, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(add_74, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_22, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(clone_20, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_23, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(add_78, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(avg_pool2d_11, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(add_81, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_24, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(clone_22, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_25, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(add_83, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_26, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_85, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(avg_pool2d_12, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_88, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_27, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_24, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_28, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_92, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(avg_pool2d_13, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_95, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_29, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_26, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_30, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_99, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(avg_pool2d_14, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_102, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_31, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_28, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_32, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_106, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(avg_pool2d_15, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_109, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_33, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_30, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_34, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_113, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(avg_pool2d_16, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_116, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_35, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_32, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_36, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_120, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(avg_pool2d_17, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_123, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_37, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_34, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_38, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_127, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(avg_pool2d_18, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_130, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_39, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_36, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_40, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_134, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(avg_pool2d_19, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_137, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_41, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_38, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_42, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_141, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(avg_pool2d_20, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_144, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_43, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_40, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_44, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_148, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(avg_pool2d_21, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_151, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_45, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_42, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_46, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_155, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(avg_pool2d_22, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_158, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_47, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_44, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_48, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_162, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(avg_pool2d_23, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_165, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_49, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_46, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_50, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_169, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(avg_pool2d_24, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_172, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_51, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_48, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_52, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_176, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(avg_pool2d_25, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_179, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_53, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_50, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_54, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_183, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(avg_pool2d_26, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_186, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_55, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_52, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_56, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_190, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(avg_pool2d_27, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_193, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_57, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_54, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_58, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_197, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(avg_pool2d_28, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_200, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_59, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_56, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_60, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_204, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(avg_pool2d_29, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_207, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_61, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_58, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_62, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(add_209, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_63, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(add_211, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(avg_pool2d_30, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(add_214, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_64, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(clone_60, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(convolution_65, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(add_218, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(avg_pool2d_31, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(add_221, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_66, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(clone_62, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(convolution_67, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(add_225, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(avg_pool2d_32, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(add_228, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_68, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(clone_64, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(convolution_69, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(add_232, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(avg_pool2d_33, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(add_235, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_70, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(clone_66, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(convolution_71, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(add_239, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(avg_pool2d_34, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(add_242, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_72, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(clone_68, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(convolution_73, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(add_246, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(avg_pool2d_35, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(add_249, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_74, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(clone_70, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(convolution_75, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(mul_324, (8, 1, 1, 768), (768, 1, 768, 1))
    assert_size_stride(view_216, (8, 768), (768, 1))
    assert_size_stride(permute_3, (1000, 768), (768, 1))
    assert_size_stride(div, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(alias_144, (8, 1), (1, 1))
    assert_size_stride(alias_145, (8, 1), (1, 1))
    assert_size_stride(alias_146, (8, 1), (1, 1))
    assert_size_stride(alias_147, (8, 1), (1, 1))
    assert_size_stride(alias_148, (8, 1), (1, 1))
    assert_size_stride(alias_149, (8, 1), (1, 1))
    assert_size_stride(alias_150, (8, 1), (1, 1))
    assert_size_stride(alias_151, (8, 1), (1, 1))
    assert_size_stride(alias_152, (8, 1), (1, 1))
    assert_size_stride(alias_153, (8, 1), (1, 1))
    assert_size_stride(alias_154, (8, 1), (1, 1))
    assert_size_stride(alias_155, (8, 1), (1, 1))
    assert_size_stride(alias_156, (8, 1), (1, 1))
    assert_size_stride(alias_157, (8, 1), (1, 1))
    assert_size_stride(alias_158, (8, 1), (1, 1))
    assert_size_stride(alias_159, (8, 1), (1, 1))
    assert_size_stride(alias_160, (8, 1), (1, 1))
    assert_size_stride(alias_161, (8, 1), (1, 1))
    assert_size_stride(alias_162, (8, 1), (1, 1))
    assert_size_stride(alias_163, (8, 1), (1, 1))
    assert_size_stride(alias_164, (8, 1), (1, 1))
    assert_size_stride(alias_165, (8, 1), (1, 1))
    assert_size_stride(alias_166, (8, 1), (1, 1))
    assert_size_stride(alias_167, (8, 1), (1, 1))
    assert_size_stride(alias_168, (8, 1), (1, 1))
    assert_size_stride(alias_169, (8, 1), (1, 1))
    assert_size_stride(alias_170, (8, 1), (1, 1))
    assert_size_stride(alias_171, (8, 1), (1, 1))
    assert_size_stride(alias_172, (8, 1), (1, 1))
    assert_size_stride(alias_173, (8, 1), (1, 1))
    assert_size_stride(alias_174, (8, 1), (1, 1))
    assert_size_stride(alias_175, (8, 1), (1, 1))
    assert_size_stride(alias_176, (8, 1), (1, 1))
    assert_size_stride(alias_177, (8, 1), (1, 1))
    assert_size_stride(alias_178, (8, 1), (1, 1))
    assert_size_stride(alias_179, (8, 1), (1, 1))
    assert_size_stride(alias_180, (8, 1), (1, 1))
    assert_size_stride(alias_181, (8, 1), (1, 1))
    assert_size_stride(alias_182, (8, 1), (1, 1))
    assert_size_stride(alias_183, (8, 1), (1, 1))
    assert_size_stride(alias_184, (8, 1), (1, 1))
    assert_size_stride(alias_185, (8, 1), (1, 1))
    assert_size_stride(alias_186, (8, 1), (1, 1))
    assert_size_stride(alias_187, (8, 1), (1, 1))
    assert_size_stride(alias_188, (8, 1), (1, 1))
    assert_size_stride(alias_189, (8, 1), (1, 1))
    assert_size_stride(alias_190, (8, 1), (1, 1))
    assert_size_stride(alias_191, (8, 1), (1, 1))
    assert_size_stride(alias_192, (8, 1), (1, 1))
    assert_size_stride(alias_193, (8, 1), (1, 1))
    assert_size_stride(alias_194, (8, 1), (1, 1))
    assert_size_stride(alias_195, (8, 1), (1, 1))
    assert_size_stride(alias_196, (8, 1), (1, 1))
    assert_size_stride(alias_197, (8, 1), (1, 1))
    assert_size_stride(alias_198, (8, 1), (1, 1))
    assert_size_stride(alias_199, (8, 1), (1, 1))
    assert_size_stride(alias_200, (8, 1), (1, 1))
    assert_size_stride(alias_201, (8, 1), (1, 1))
    assert_size_stride(alias_202, (8, 1), (1, 1))
    assert_size_stride(alias_203, (8, 1), (1, 1))
    assert_size_stride(alias_204, (8, 1), (1, 1))
    assert_size_stride(alias_205, (8, 1), (1, 1))
    assert_size_stride(alias_206, (8, 1), (1, 1))
    assert_size_stride(alias_207, (8, 1), (1, 1))
    assert_size_stride(alias_208, (8, 1), (1, 1))
    assert_size_stride(alias_209, (8, 1), (1, 1))
    assert_size_stride(alias_210, (8, 1), (1, 1))
    assert_size_stride(alias_211, (8, 1), (1, 1))
    assert_size_stride(alias_212, (8, 1), (1, 1))
    assert_size_stride(alias_213, (8, 1), (1, 1))
    assert_size_stride(alias_214, (8, 1), (1, 1))
    assert_size_stride(alias_215, (8, 1), (1, 1))
    assert_size_stride(alias_216, (8, 1), (1, 1))
    assert_size_stride(alias_217, (8, 1), (1, 1))
    assert_size_stride(alias_218, (8, 1), (1, 1))
    assert_size_stride(alias_219, (8, 1), (1, 1))
    assert_size_stride(alias_220, (8, 1), (1, 1))
    assert_size_stride(alias_221, (8, 1), (1, 1))
    assert_size_stride(alias_222, (8, 1), (1, 1))
    assert_size_stride(alias_223, (8, 1), (1, 1))
    assert_size_stride(alias_224, (8, 1), (1, 1))
    assert_size_stride(alias_225, (8, 1), (1, 1))
    assert_size_stride(alias_226, (8, 1), (1, 1))
    assert_size_stride(alias_227, (8, 1), (1, 1))
    assert_size_stride(alias_228, (8, 1), (1, 1))
    assert_size_stride(alias_229, (8, 1), (1, 1))
    assert_size_stride(alias_230, (8, 1), (1, 1))
    assert_size_stride(alias_231, (8, 1), (1, 1))
    assert_size_stride(alias_232, (8, 1), (1, 1))
    assert_size_stride(alias_233, (8, 1), (1, 1))
    assert_size_stride(alias_234, (8, 1), (1, 1))
    assert_size_stride(alias_235, (8, 1), (1, 1))
    assert_size_stride(alias_236, (8, 1), (1, 1))
    assert_size_stride(alias_237, (8, 1), (1, 1))
    assert_size_stride(alias_238, (8, 1), (1, 1))
    assert_size_stride(alias_239, (8, 1), (1, 1))
    assert_size_stride(alias_240, (8, 1), (1, 1))
    assert_size_stride(alias_241, (8, 1), (1, 1))
    assert_size_stride(alias_242, (8, 1), (1, 1))
    assert_size_stride(alias_243, (8, 1), (1, 1))
    assert_size_stride(alias_244, (8, 1), (1, 1))
    assert_size_stride(alias_245, (8, 1), (1, 1))
    assert_size_stride(alias_246, (8, 1), (1, 1))
    assert_size_stride(alias_247, (8, 1), (1, 1))
    assert_size_stride(alias_248, (8, 1), (1, 1))
    assert_size_stride(alias_249, (8, 1), (1, 1))
    assert_size_stride(alias_250, (8, 1), (1, 1))
    assert_size_stride(alias_251, (8, 1), (1, 1))
    assert_size_stride(alias_252, (8, 1), (1, 1))
    assert_size_stride(alias_253, (8, 1), (1, 1))
    assert_size_stride(alias_254, (8, 1), (1, 1))
    assert_size_stride(alias_255, (8, 1), (1, 1))
    assert_size_stride(alias_256, (8, 1), (1, 1))
    assert_size_stride(alias_257, (8, 1), (1, 1))
    assert_size_stride(alias_258, (8, 1), (1, 1))
    assert_size_stride(alias_259, (8, 1), (1, 1))
    assert_size_stride(alias_260, (8, 1), (1, 1))
    assert_size_stride(alias_261, (8, 1), (1, 1))
    assert_size_stride(alias_262, (8, 1), (1, 1))
    assert_size_stride(alias_263, (8, 1), (1, 1))
    assert_size_stride(alias_264, (8, 1), (1, 1))
    assert_size_stride(alias_265, (8, 1), (1, 1))
    assert_size_stride(alias_266, (8, 1), (1, 1))
    assert_size_stride(alias_267, (8, 1), (1, 1))
    assert_size_stride(alias_268, (8, 1), (1, 1))
    assert_size_stride(alias_269, (8, 1), (1, 1))
    assert_size_stride(alias_270, (8, 1), (1, 1))
    assert_size_stride(alias_271, (8, 1), (1, 1))
    assert_size_stride(alias_272, (8, 1), (1, 1))
    assert_size_stride(alias_273, (8, 1), (1, 1))
    assert_size_stride(alias_274, (8, 1), (1, 1))
    assert_size_stride(alias_275, (8, 1), (1, 1))
    assert_size_stride(alias_276, (8, 1), (1, 1))
    assert_size_stride(alias_277, (8, 1), (1, 1))
    assert_size_stride(alias_278, (8, 1), (1, 1))
    assert_size_stride(alias_279, (8, 1), (1, 1))
    assert_size_stride(alias_280, (8, 1), (1, 1))
    assert_size_stride(alias_281, (8, 1), (1, 1))
    assert_size_stride(alias_282, (8, 1), (1, 1))
    assert_size_stride(alias_283, (8, 1), (1, 1))
    assert_size_stride(alias_284, (8, 1), (1, 1))
    assert_size_stride(alias_285, (8, 1), (1, 1))
    assert_size_stride(alias_286, (8, 1), (1, 1))
    assert_size_stride(alias_287, (8, 1), (1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 96, 56, 56), device='cuda', dtype=torch.float32)
        buf1 = empty((8, 96, 56, 56), device='cuda', dtype=torch.float32)
        buf2 = empty((8, 96, 56, 56), device='cuda', dtype=torch.float32)
        buf3 = empty((8, 96, 56, 56), device='cuda', dtype=torch.float32)
        buf4 = empty((8, 96, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul, mul_1, mul_2, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, sub, sub_1, sub_2, sub_3, sub_4, x_11, x_12, x_19, x_20, x_27, x_28, x_35, x_36, x_4, x_43], Original ATen: [aten.add, aten.mul, aten.sub]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_add_mul_sub_0.run(convolution, avg_pool2d, add_1, primals_3, convolution_2, primals_6, avg_pool2d_1, add_8, primals_9, convolution_4, primals_12, avg_pool2d_2, add_15, primals_15, convolution_6, primals_18, avg_pool2d_3, add_22, primals_21, convolution_8, primals_24, avg_pool2d_4, add_29, primals_27, convolution_10, primals_30, buf0, buf1, buf2, buf3, buf4, 25088, 96, grid=grid(25088, 96), stream=stream0)
        buf5 = empty((8, 192, 28, 28), device='cuda', dtype=torch.float32)
        buf6 = empty((8, 192, 28, 28), device='cuda', dtype=torch.float32)
        buf7 = empty((8, 192, 28, 28), device='cuda', dtype=torch.float32)
        buf8 = empty((8, 192, 28, 28), device='cuda', dtype=torch.float32)
        buf9 = empty((8, 192, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_12, mul_13, mul_14, mul_15, mul_16, mul_17, mul_18, mul_19, mul_20, mul_21, sub_10, sub_6, sub_7, sub_8, sub_9, x_56, x_63, x_64, x_71, x_72, x_79, x_80, x_87, x_88, x_95], Original ATen: [aten.add, aten.mul, aten.sub]
        triton_poi_fused_add_mul_sub_1.run(convolution_13, avg_pool2d_6, add_43, primals_39, convolution_15, primals_42, avg_pool2d_7, add_50, primals_45, convolution_17, primals_48, avg_pool2d_8, add_57, primals_51, convolution_19, primals_54, avg_pool2d_9, add_64, primals_57, convolution_21, primals_60, avg_pool2d_10, add_71, primals_63, convolution_23, primals_66, buf5, buf6, buf7, buf8, buf9, 6272, 192, grid=grid(6272, 192), stream=stream0)
        buf10 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf11 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf12 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf13 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf14 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf15 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf16 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf17 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf18 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf19 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf20 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf21 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf22 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf23 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf24 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf25 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf26 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_24, mul_25, mul_26, mul_27, mul_28, mul_29, mul_30, mul_31, mul_32, mul_33, mul_34, mul_35, mul_36, mul_37, mul_38, mul_39, mul_40, mul_41, mul_42, mul_43, mul_44, mul_45, mul_46, mul_47, mul_48, mul_49, mul_50, mul_51, mul_52, mul_53, mul_54, mul_55, mul_56, mul_57, sub_12, sub_13, sub_14, sub_15, sub_16, sub_17, sub_18, sub_19, sub_20, sub_21, sub_22, sub_23, sub_24, sub_25, sub_26, sub_27, sub_28, x_108, x_115, x_116, x_123, x_124, x_131, x_132, x_139, x_140, x_147, x_148, x_155, x_156, x_163, x_164, x_171, x_172, x_179, x_180, x_187, x_188, x_195, x_196, x_203, x_204, x_211, x_212, x_219, x_220, x_227, x_228, x_235, x_236, x_243], Original ATen: [aten.add, aten.mul, aten.sub]
        triton_poi_fused_add_mul_sub_2.run(convolution_26, avg_pool2d_12, add_85, primals_75, convolution_28, primals_78, avg_pool2d_13, add_92, primals_81, convolution_30, primals_84, avg_pool2d_14, add_99, primals_87, convolution_32, primals_90, avg_pool2d_15, add_106, primals_93, convolution_34, primals_96, avg_pool2d_16, add_113, primals_99, convolution_36, primals_102, avg_pool2d_17, add_120, primals_105, convolution_38, primals_108, avg_pool2d_18, add_127, primals_111, convolution_40, primals_114, avg_pool2d_19, add_134, primals_117, convolution_42, primals_120, avg_pool2d_20, add_141, primals_123, convolution_44, primals_126, avg_pool2d_21, add_148, primals_129, convolution_46, primals_132, avg_pool2d_22, add_155, primals_135, convolution_48, primals_138, avg_pool2d_23, add_162, primals_141, convolution_50, primals_144, avg_pool2d_24, add_169, primals_147, convolution_52, primals_150, avg_pool2d_25, add_176, primals_153, convolution_54, primals_156, avg_pool2d_26, add_183, primals_159, convolution_56, primals_162, avg_pool2d_27, add_190, primals_165, convolution_58, primals_168, avg_pool2d_28, add_197, primals_171, convolution_60, primals_174, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf27 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        buf28 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        buf29 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        buf30 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        buf31 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_60, mul_61, mul_62, mul_63, mul_64, mul_65, mul_66, mul_67, mul_68, mul_69, sub_30, sub_31, sub_32, sub_33, sub_34, x_256, x_263, x_264, x_271, x_272, x_279, x_280, x_287, x_288, x_295], Original ATen: [aten.add, aten.mul, aten.sub]
        triton_poi_fused_add_mul_sub_3.run(convolution_63, avg_pool2d_30, add_211, primals_183, convolution_65, primals_186, avg_pool2d_31, add_218, primals_189, convolution_67, primals_192, avg_pool2d_32, add_225, primals_195, convolution_69, primals_198, avg_pool2d_33, add_232, primals_201, convolution_71, primals_204, avg_pool2d_34, add_239, primals_207, convolution_73, primals_210, buf27, buf28, buf29, buf30, buf31, 392, 768, grid=grid(392, 768), stream=stream0)
        buf32 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_3, out=buf32)
        del permute_3
        buf33 = empty((1000, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), view_216, out=buf33)
        del view_216
        buf34 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(tangents_1, buf34, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf39 = empty_strided((8, 1, 1, 768), (768, 6144, 6144, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf32, primals_217, mul_324, div, buf39, 8, 768, grid=grid(8), stream=stream0)
        del div
        del primals_217
        buf37 = empty((768, ), device='cuda', dtype=torch.float32)
        buf38 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_6.run(buf32, mul_324, buf37, buf38, 768, 8, grid=grid(768), stream=stream0)
        del mul_324
        buf40 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul]
        triton_poi_fused_div_mul_7.run(buf39, primals_216, buf40, 301056, grid=grid(301056), stream=stream0)
        del primals_216
        buf41 = empty_strided((1, 768, 1, 1, 4), (3072, 1, 3072, 3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.sum]
        triton_red_fused_div_mul_sum_8.run(buf39, convolution_75, buf41, 3072, 98, grid=grid(3072), stream=stream0)
        del convolution_75
        buf42 = empty((1, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.sum]
        triton_per_fused_div_mul_sum_9.run(buf41, buf42, 768, 4, grid=grid(768), stream=stream0)
        buf43 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf40, buf43, 768, 392, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf44 = aten.convolution_backward(buf40, clone_70, primals_369, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_70
        del primals_369
        buf45 = buf44[0]
        buf46 = buf44[1]
        del buf44
        buf47 = buf45; del buf45  # reuse
        # Source Nodes: [x_298], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_11.run(buf47, convolution_74, 24576, 49, grid=grid(24576, 49), stream=stream0)
        del convolution_74
        buf48 = reinterpret_tensor(buf41, (3072, ), (1, ), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf47, buf48, 3072, 392, grid=grid(3072), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf49 = aten.convolution_backward(buf47, add_249, primals_367, [3072], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_249
        del buf47
        del primals_367
        buf50 = buf49[0]
        buf51 = buf49[1]
        del buf49
        buf52 = buf32; del buf32  # reuse
        buf53 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_13.run(buf50, buf31, avg_pool2d_35, add_246, primals_213, buf52, buf53, 6144, 49, grid=grid(6144), stream=stream0)
        buf54 = empty_strided((8, 1), (1, 8), device='cuda', dtype=torch.float32)
        buf55 = empty_strided((8, 1), (1, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_14.run(buf52, primals_214, buf53, buf54, buf55, 8, 768, grid=grid(8), stream=stream0)
        buf56 = reinterpret_tensor(buf50, (8, 1, 768, 49), (37632, 301056, 49, 1), 0); del buf50  # reuse
        buf59 = reinterpret_tensor(buf56, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf56  # reuse
        buf60 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_div_mul_native_group_norm_backward_15.run(buf59, alias_145, primals_214, buf31, avg_pool2d_35, add_246, primals_213, buf55, alias_144, buf54, buf39, buf60, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del buf39
        del primals_213
        del primals_214
        buf57 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf58 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_16.run(buf52, buf53, alias_144, alias_145, buf57, buf58, 768, 8, grid=grid(768), stream=stream0)
        del alias_144
        del alias_145
        buf61 = empty_strided((1, 768, 1, 1, 4), (3072, 1, 3072, 3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_35], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_17.run(buf59, avg_pool2d_35, add_246, buf61, 3072, 98, grid=grid(3072), stream=stream0)
        del add_246
        del avg_pool2d_35
        buf62 = empty((1, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_35], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_div_mul_sum_9.run(buf61, buf62, 768, 4, grid=grid(768), stream=stream0)
        buf63 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        buf64 = buf53; del buf53  # reuse
        buf65 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_18.run(buf60, buf31, buf63, buf64, buf65, 6144, 49, grid=grid(6144), stream=stream0)
        buf66 = buf55; del buf55  # reuse
        buf67 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_14.run(buf64, primals_211, buf65, buf66, buf67, 8, 768, grid=grid(8), stream=stream0)
        buf68 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf69 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_16.run(buf64, buf65, alias_146, alias_147, buf68, buf69, 768, 8, grid=grid(768), stream=stream0)
        buf70 = buf31; del buf31  # reuse
        buf71 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf70, buf59, buf60, buf63, alias_147, primals_211, buf67, alias_146, buf66, primals_210, buf71, 301056, grid=grid(301056), stream=stream0)
        del alias_146
        del alias_147
        del buf59
        del buf60
        del primals_210
        del primals_211
        buf72 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_20.run(buf70, convolution_73, buf72, 3072, 98, grid=grid(3072), stream=stream0)
        del convolution_73
        buf73 = empty((1, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_sum_9.run(buf72, buf73, 768, 4, grid=grid(768), stream=stream0)
        buf74 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf71, buf74, 768, 392, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf75 = aten.convolution_backward(buf71, clone_68, primals_365, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_68
        del primals_365
        buf76 = buf75[0]
        buf77 = buf75[1]
        del buf75
        buf78 = buf76; del buf76  # reuse
        # Source Nodes: [x_290], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_11.run(buf78, convolution_72, 24576, 49, grid=grid(24576, 49), stream=stream0)
        del convolution_72
        buf79 = reinterpret_tensor(buf72, (3072, ), (1, ), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf78, buf79, 3072, 392, grid=grid(3072), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf80 = aten.convolution_backward(buf78, add_242, primals_363, [3072], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_242
        del buf78
        del primals_363
        buf81 = buf80[0]
        buf82 = buf80[1]
        del buf80
        buf83 = buf65; del buf65  # reuse
        buf84 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_13.run(buf81, buf30, avg_pool2d_34, add_239, primals_207, buf83, buf84, 6144, 49, grid=grid(6144), stream=stream0)
        buf85 = buf67; del buf67  # reuse
        buf86 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_14.run(buf83, primals_208, buf84, buf85, buf86, 8, 768, grid=grid(8), stream=stream0)
        buf87 = reinterpret_tensor(buf81, (8, 1, 768, 49), (37632, 301056, 49, 1), 0); del buf81  # reuse
        buf90 = buf70; del buf70  # reuse
        buf91 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_21.run(buf87, buf90, alias_149, primals_208, buf30, avg_pool2d_34, add_239, primals_207, buf86, alias_148, buf85, buf91, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_207
        del primals_208
        buf88 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf89 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_16.run(buf83, buf84, alias_148, alias_149, buf88, buf89, 768, 8, grid=grid(768), stream=stream0)
        del alias_148
        del alias_149
        buf92 = empty_strided((1, 768, 1, 1, 4), (3072, 1, 3072, 3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_34], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_17.run(buf90, avg_pool2d_34, add_239, buf92, 3072, 98, grid=grid(3072), stream=stream0)
        del add_239
        del avg_pool2d_34
        buf93 = empty((1, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_34], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_div_mul_sum_9.run(buf92, buf93, 768, 4, grid=grid(768), stream=stream0)
        buf94 = reinterpret_tensor(buf87, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf87  # reuse
        buf95 = buf84; del buf84  # reuse
        buf96 = buf83; del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_18.run(buf91, buf30, buf94, buf95, buf96, 6144, 49, grid=grid(6144), stream=stream0)
        buf97 = buf86; del buf86  # reuse
        buf98 = buf85; del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_14.run(buf95, primals_205, buf96, buf97, buf98, 8, 768, grid=grid(8), stream=stream0)
        buf99 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf100 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_16.run(buf95, buf96, alias_150, alias_151, buf99, buf100, 768, 8, grid=grid(768), stream=stream0)
        buf101 = buf30; del buf30  # reuse
        buf102 = buf63; del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_19.run(buf101, buf90, buf91, buf94, alias_151, primals_205, buf98, alias_150, buf97, primals_204, buf102, 301056, grid=grid(301056), stream=stream0)
        del alias_150
        del alias_151
        del buf90
        del buf91
        del primals_204
        del primals_205
        buf103 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_20.run(buf101, convolution_71, buf103, 3072, 98, grid=grid(3072), stream=stream0)
        del convolution_71
        buf104 = empty((1, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_sum_9.run(buf103, buf104, 768, 4, grid=grid(768), stream=stream0)
        buf105 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf102, buf105, 768, 392, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf106 = aten.convolution_backward(buf102, clone_66, primals_361, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_66
        del primals_361
        buf107 = buf106[0]
        buf108 = buf106[1]
        del buf106
        buf109 = buf107; del buf107  # reuse
        # Source Nodes: [x_282], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_11.run(buf109, convolution_70, 24576, 49, grid=grid(24576, 49), stream=stream0)
        del convolution_70
        buf110 = reinterpret_tensor(buf103, (3072, ), (1, ), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf109, buf110, 3072, 392, grid=grid(3072), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf111 = aten.convolution_backward(buf109, add_235, primals_359, [3072], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_235
        del buf109
        del primals_359
        buf112 = buf111[0]
        buf113 = buf111[1]
        del buf111
        buf114 = buf96; del buf96  # reuse
        buf115 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_13.run(buf112, buf29, avg_pool2d_33, add_232, primals_201, buf114, buf115, 6144, 49, grid=grid(6144), stream=stream0)
        buf116 = buf98; del buf98  # reuse
        buf117 = buf97; del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_14.run(buf114, primals_202, buf115, buf116, buf117, 8, 768, grid=grid(8), stream=stream0)
        buf118 = reinterpret_tensor(buf112, (8, 1, 768, 49), (37632, 301056, 49, 1), 0); del buf112  # reuse
        buf121 = buf101; del buf101  # reuse
        buf122 = buf102; del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_21.run(buf118, buf121, alias_153, primals_202, buf29, avg_pool2d_33, add_232, primals_201, buf117, alias_152, buf116, buf122, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_201
        del primals_202
        buf119 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf120 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_16.run(buf114, buf115, alias_152, alias_153, buf119, buf120, 768, 8, grid=grid(768), stream=stream0)
        del alias_152
        del alias_153
        buf123 = empty_strided((1, 768, 1, 1, 4), (3072, 1, 3072, 3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_33], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_17.run(buf121, avg_pool2d_33, add_232, buf123, 3072, 98, grid=grid(3072), stream=stream0)
        del add_232
        del avg_pool2d_33
        buf124 = empty((1, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_33], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_div_mul_sum_9.run(buf123, buf124, 768, 4, grid=grid(768), stream=stream0)
        buf125 = reinterpret_tensor(buf118, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf118  # reuse
        buf126 = buf115; del buf115  # reuse
        buf127 = buf114; del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_18.run(buf122, buf29, buf125, buf126, buf127, 6144, 49, grid=grid(6144), stream=stream0)
        buf128 = buf117; del buf117  # reuse
        buf129 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_14.run(buf126, primals_199, buf127, buf128, buf129, 8, 768, grid=grid(8), stream=stream0)
        buf130 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf131 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_16.run(buf126, buf127, alias_154, alias_155, buf130, buf131, 768, 8, grid=grid(768), stream=stream0)
        buf132 = buf121; del buf121  # reuse
        buf133 = buf94; del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf132, buf122, buf125, alias_155, primals_199, buf29, buf129, alias_154, buf128, primals_198, buf133, 301056, grid=grid(301056), stream=stream0)
        del alias_154
        del alias_155
        del buf122
        del buf125
        del primals_198
        del primals_199
        buf134 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_20.run(buf132, convolution_69, buf134, 3072, 98, grid=grid(3072), stream=stream0)
        del convolution_69
        buf135 = empty((1, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_sum_9.run(buf134, buf135, 768, 4, grid=grid(768), stream=stream0)
        buf136 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf133, buf136, 768, 392, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf137 = aten.convolution_backward(buf133, clone_64, primals_357, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_64
        del primals_357
        buf138 = buf137[0]
        buf139 = buf137[1]
        del buf137
        buf140 = buf138; del buf138  # reuse
        # Source Nodes: [x_274], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_11.run(buf140, convolution_68, 24576, 49, grid=grid(24576, 49), stream=stream0)
        del convolution_68
        buf141 = reinterpret_tensor(buf134, (3072, ), (1, ), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf140, buf141, 3072, 392, grid=grid(3072), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf142 = aten.convolution_backward(buf140, add_228, primals_355, [3072], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_228
        del buf140
        del primals_355
        buf143 = buf142[0]
        buf144 = buf142[1]
        del buf142
        buf145 = buf127; del buf127  # reuse
        buf146 = buf126; del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_13.run(buf143, buf28, avg_pool2d_32, add_225, primals_195, buf145, buf146, 6144, 49, grid=grid(6144), stream=stream0)
        buf147 = buf129; del buf129  # reuse
        buf148 = buf128; del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_14.run(buf145, primals_196, buf146, buf147, buf148, 8, 768, grid=grid(8), stream=stream0)
        buf149 = reinterpret_tensor(buf143, (8, 1, 768, 49), (37632, 301056, 49, 1), 0); del buf143  # reuse
        buf152 = buf132; del buf132  # reuse
        buf153 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_21.run(buf149, buf152, alias_157, primals_196, buf28, avg_pool2d_32, add_225, primals_195, buf148, alias_156, buf147, buf153, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_195
        del primals_196
        buf150 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf151 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_16.run(buf145, buf146, alias_156, alias_157, buf150, buf151, 768, 8, grid=grid(768), stream=stream0)
        del alias_156
        del alias_157
        buf154 = empty_strided((1, 768, 1, 1, 4), (3072, 1, 3072, 3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_32], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_17.run(buf152, avg_pool2d_32, add_225, buf154, 3072, 98, grid=grid(3072), stream=stream0)
        del add_225
        del avg_pool2d_32
        buf155 = empty((1, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_32], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_div_mul_sum_9.run(buf154, buf155, 768, 4, grid=grid(768), stream=stream0)
        buf156 = reinterpret_tensor(buf149, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf149  # reuse
        buf157 = buf146; del buf146  # reuse
        buf158 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_18.run(buf153, buf28, buf156, buf157, buf158, 6144, 49, grid=grid(6144), stream=stream0)
        buf159 = buf148; del buf148  # reuse
        buf160 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_14.run(buf157, primals_193, buf158, buf159, buf160, 8, 768, grid=grid(8), stream=stream0)
        buf161 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf162 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_16.run(buf157, buf158, alias_158, alias_159, buf161, buf162, 768, 8, grid=grid(768), stream=stream0)
        buf163 = buf152; del buf152  # reuse
        buf164 = buf29; del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf163, buf153, buf156, alias_159, primals_193, buf28, buf160, alias_158, buf159, primals_192, buf164, 301056, grid=grid(301056), stream=stream0)
        del alias_158
        del alias_159
        del buf153
        del buf156
        del primals_192
        del primals_193
        buf165 = buf154; del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_20.run(buf163, convolution_67, buf165, 3072, 98, grid=grid(3072), stream=stream0)
        del convolution_67
        buf166 = empty((1, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_sum_9.run(buf165, buf166, 768, 4, grid=grid(768), stream=stream0)
        buf167 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf164, buf167, 768, 392, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf168 = aten.convolution_backward(buf164, clone_62, primals_353, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_62
        del primals_353
        buf169 = buf168[0]
        buf170 = buf168[1]
        del buf168
        buf171 = buf169; del buf169  # reuse
        # Source Nodes: [x_266], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_11.run(buf171, convolution_66, 24576, 49, grid=grid(24576, 49), stream=stream0)
        del convolution_66
        buf172 = reinterpret_tensor(buf165, (3072, ), (1, ), 0); del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf171, buf172, 3072, 392, grid=grid(3072), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf173 = aten.convolution_backward(buf171, add_221, primals_351, [3072], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_221
        del primals_351
        buf174 = buf173[0]
        buf175 = buf173[1]
        del buf173
        buf176 = buf158; del buf158  # reuse
        buf177 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_13.run(buf174, buf27, avg_pool2d_31, add_218, primals_189, buf176, buf177, 6144, 49, grid=grid(6144), stream=stream0)
        buf178 = buf160; del buf160  # reuse
        buf179 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_14.run(buf176, primals_190, buf177, buf178, buf179, 8, 768, grid=grid(8), stream=stream0)
        buf180 = reinterpret_tensor(buf174, (8, 1, 768, 49), (37632, 301056, 49, 1), 0); del buf174  # reuse
        buf183 = buf163; del buf163  # reuse
        buf184 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_21.run(buf180, buf183, alias_161, primals_190, buf27, avg_pool2d_31, add_218, primals_189, buf179, alias_160, buf178, buf184, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del primals_189
        del primals_190
        buf181 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf182 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_16.run(buf176, buf177, alias_160, alias_161, buf181, buf182, 768, 8, grid=grid(768), stream=stream0)
        del alias_160
        del alias_161
        buf185 = empty_strided((1, 768, 1, 1, 4), (3072, 1, 3072, 3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_31], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_17.run(buf183, avg_pool2d_31, add_218, buf185, 3072, 98, grid=grid(3072), stream=stream0)
        del add_218
        del avg_pool2d_31
        buf186 = empty((1, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_31], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_div_mul_sum_9.run(buf185, buf186, 768, 4, grid=grid(768), stream=stream0)
        buf187 = reinterpret_tensor(buf180, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf180  # reuse
        buf188 = buf177; del buf177  # reuse
        buf189 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_18.run(buf184, buf27, buf187, buf188, buf189, 6144, 49, grid=grid(6144), stream=stream0)
        buf190 = buf179; del buf179  # reuse
        buf191 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_14.run(buf188, primals_187, buf189, buf190, buf191, 8, 768, grid=grid(8), stream=stream0)
        buf192 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf193 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_16.run(buf188, buf189, alias_162, alias_163, buf192, buf193, 768, 8, grid=grid(768), stream=stream0)
        buf194 = buf183; del buf183  # reuse
        buf195 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf194, buf184, buf187, alias_163, primals_187, buf27, buf191, alias_162, buf190, primals_186, buf195, 301056, grid=grid(301056), stream=stream0)
        del alias_162
        del alias_163
        del buf184
        del buf187
        del buf27
        del primals_186
        del primals_187
        buf196 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_20.run(buf194, convolution_65, buf196, 3072, 98, grid=grid(3072), stream=stream0)
        del convolution_65
        buf197 = empty((1, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_sum_9.run(buf196, buf197, 768, 4, grid=grid(768), stream=stream0)
        buf198 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf195, buf198, 768, 392, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf199 = aten.convolution_backward(buf195, clone_60, primals_349, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_60
        del primals_349
        buf200 = buf199[0]
        buf201 = buf199[1]
        del buf199
        buf202 = buf200; del buf200  # reuse
        # Source Nodes: [x_258], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_11.run(buf202, convolution_64, 24576, 49, grid=grid(24576, 49), stream=stream0)
        del convolution_64
        buf203 = reinterpret_tensor(buf196, (3072, ), (1, ), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf202, buf203, 3072, 392, grid=grid(3072), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf204 = aten.convolution_backward(buf202, add_214, primals_347, [3072], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_214
        del primals_347
        buf205 = buf204[0]
        buf206 = buf204[1]
        del buf204
        buf207 = buf189; del buf189  # reuse
        buf208 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_23.run(buf205, convolution_63, avg_pool2d_30, add_211, primals_183, buf207, buf208, 6144, 49, grid=grid(6144), stream=stream0)
        buf209 = buf191; del buf191  # reuse
        buf210 = buf190; del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_14.run(buf207, primals_184, buf208, buf209, buf210, 8, 768, grid=grid(8), stream=stream0)
        buf211 = reinterpret_tensor(buf195, (8, 1, 768, 49), (37632, 301056, 1, 768), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_poi_fused_native_group_norm_backward_24.run(buf205, alias_165, primals_184, convolution_63, avg_pool2d_30, add_211, primals_183, buf210, alias_164, buf209, buf211, 392, 768, grid=grid(392, 768), stream=stream0)
        del primals_184
        buf212 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf213 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_16.run(buf207, buf208, alias_164, alias_165, buf212, buf213, 768, 8, grid=grid(768), stream=stream0)
        buf214 = buf194; del buf194  # reuse
        buf215 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf214, buf211, buf210, alias_164, buf209, alias_165, primals_183, buf215, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del alias_164
        del alias_165
        del primals_183
        buf216 = empty_strided((1, 768, 1, 1, 4), (3072, 1, 3072, 3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_30], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_17.run(buf214, avg_pool2d_30, add_211, buf216, 3072, 98, grid=grid(3072), stream=stream0)
        del add_211
        del avg_pool2d_30
        buf217 = empty((1, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_30], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_div_mul_sum_9.run(buf216, buf217, 768, 4, grid=grid(768), stream=stream0)
        buf218 = reinterpret_tensor(buf211, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf211  # reuse
        buf219 = buf208; del buf208  # reuse
        buf220 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_26.run(buf215, convolution_63, buf218, buf219, buf220, 6144, 49, grid=grid(6144), stream=stream0)
        buf221 = buf210; del buf210  # reuse
        buf222 = buf209; del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_14.run(buf219, primals_181, buf220, buf221, buf222, 8, 768, grid=grid(8), stream=stream0)
        buf223 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf224 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_16.run(buf219, buf220, alias_166, alias_167, buf223, buf224, 768, 8, grid=grid(768), stream=stream0)
        del buf219
        buf225 = buf214; del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_27.run(buf225, buf215, buf218, alias_167, primals_181, convolution_63, buf222, alias_166, buf221, 6144, 49, grid=grid(6144, 49), stream=stream0)
        del alias_166
        del alias_167
        del buf215
        del buf218
        del convolution_63
        del primals_181
        buf226 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf225, buf226, 768, 392, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf227 = aten.convolution_backward(buf225, add_209, primals_345, [768], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_209
        del buf225
        del primals_345
        buf228 = buf227[0]
        buf229 = buf227[1]
        del buf227
        buf230 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_28.run(buf228, primals_180, buf230, 602112, grid=grid(602112), stream=stream0)
        del primals_180
        buf231 = empty_strided((1, 384, 1, 1, 13), (4992, 13, 4992, 4992, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29.run(buf228, convolution_62, buf231, 4992, 121, grid=grid(4992), stream=stream0)
        del convolution_62
        buf232 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_30.run(buf231, buf232, 384, 13, grid=grid(384), stream=stream0)
        buf233 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf230, buf233, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf234 = aten.convolution_backward(buf230, clone_58, primals_343, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_58
        del primals_343
        buf235 = buf234[0]
        buf236 = buf234[1]
        del buf234
        buf237 = buf235; del buf235  # reuse
        # Source Nodes: [x_246], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf237, convolution_61, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del convolution_61
        buf238 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf237, buf238, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf239 = aten.convolution_backward(buf237, add_207, primals_341, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_207
        del buf237
        del primals_341
        buf240 = buf239[0]
        buf241 = buf239[1]
        del buf239
        buf242 = reinterpret_tensor(buf220, (8, 384, 2), (768, 2, 1), 0); del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_34.run(buf240, buf26, avg_pool2d_29, add_204, primals_177, buf242, 6144, 98, grid=grid(6144), stream=stream0)
        buf243 = reinterpret_tensor(buf216, (8, 384), (384, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_35.run(buf242, buf243, 3072, 2, grid=grid(3072), stream=stream0)
        buf244 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_36.run(buf240, buf244, 3072, 196, grid=grid(3072), stream=stream0)
        buf245 = buf222; del buf222  # reuse
        buf246 = buf221; del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf243, primals_178, buf244, buf245, buf246, 8, 384, grid=grid(8), stream=stream0)
        buf247 = reinterpret_tensor(buf240, (8, 1, 384, 196), (75264, 602112, 196, 1), 0); del buf240  # reuse
        buf250 = buf228; del buf228  # reuse
        buf251 = buf230; del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_38.run(buf247, buf250, alias_169, primals_178, buf26, avg_pool2d_29, add_204, primals_177, buf246, alias_168, buf245, buf251, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_177
        del primals_178
        buf248 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf249 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf243, buf244, alias_168, alias_169, buf248, buf249, 384, 8, grid=grid(384), stream=stream0)
        del alias_168
        del alias_169
        buf252 = reinterpret_tensor(buf231, (1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), 0); del buf231  # reuse
        # Source Nodes: [sub_29], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_40.run(buf250, avg_pool2d_29, add_204, buf252, 4992, 121, grid=grid(4992), stream=stream0)
        del add_204
        del avg_pool2d_29
        buf253 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_29], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_41.run(buf252, buf253, 384, 13, grid=grid(384), stream=stream0)
        buf254 = reinterpret_tensor(buf247, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf247  # reuse
        buf255 = buf244; del buf244  # reuse
        buf256 = buf243; del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42.run(buf251, buf26, buf254, buf255, buf256, 3072, 196, grid=grid(3072), stream=stream0)
        buf257 = buf246; del buf246  # reuse
        buf258 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf255, primals_175, buf256, buf257, buf258, 8, 384, grid=grid(8), stream=stream0)
        buf259 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf260 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf255, buf256, alias_170, alias_171, buf259, buf260, 384, 8, grid=grid(384), stream=stream0)
        buf261 = buf250; del buf250  # reuse
        buf262 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_43.run(buf261, buf251, buf254, alias_171, primals_175, buf26, buf258, alias_170, buf257, primals_174, buf262, 602112, grid=grid(602112), stream=stream0)
        del alias_170
        del alias_171
        del buf251
        del buf254
        del primals_174
        del primals_175
        buf263 = reinterpret_tensor(buf252, (1, 384, 1, 1, 13), (4992, 13, 4992, 4992, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29.run(buf261, convolution_60, buf263, 4992, 121, grid=grid(4992), stream=stream0)
        del convolution_60
        buf264 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_30.run(buf263, buf264, 384, 13, grid=grid(384), stream=stream0)
        buf265 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf262, buf265, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf266 = aten.convolution_backward(buf262, clone_56, primals_339, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_56
        del primals_339
        buf267 = buf266[0]
        buf268 = buf266[1]
        del buf266
        buf269 = buf267; del buf267  # reuse
        # Source Nodes: [x_238], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf269, convolution_59, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del convolution_59
        buf270 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf269, buf270, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf271 = aten.convolution_backward(buf269, add_200, primals_337, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_200
        del buf269
        del primals_337
        buf272 = buf271[0]
        buf273 = buf271[1]
        del buf271
        buf274 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_34.run(buf272, buf25, avg_pool2d_28, add_197, primals_171, buf274, 6144, 98, grid=grid(6144), stream=stream0)
        buf275 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_35.run(buf274, buf275, 3072, 2, grid=grid(3072), stream=stream0)
        buf276 = buf255; del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_36.run(buf272, buf276, 3072, 196, grid=grid(3072), stream=stream0)
        buf277 = buf258; del buf258  # reuse
        buf278 = buf257; del buf257  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf275, primals_172, buf276, buf277, buf278, 8, 384, grid=grid(8), stream=stream0)
        buf279 = reinterpret_tensor(buf272, (8, 1, 384, 196), (75264, 602112, 196, 1), 0); del buf272  # reuse
        buf282 = buf261; del buf261  # reuse
        buf283 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_38.run(buf279, buf282, alias_173, primals_172, buf25, avg_pool2d_28, add_197, primals_171, buf278, alias_172, buf277, buf283, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_171
        del primals_172
        buf280 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf281 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf275, buf276, alias_172, alias_173, buf280, buf281, 384, 8, grid=grid(384), stream=stream0)
        del alias_172
        del alias_173
        buf284 = reinterpret_tensor(buf263, (1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), 0); del buf263  # reuse
        # Source Nodes: [sub_28], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_40.run(buf282, avg_pool2d_28, add_197, buf284, 4992, 121, grid=grid(4992), stream=stream0)
        del add_197
        del avg_pool2d_28
        buf285 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_28], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_41.run(buf284, buf285, 384, 13, grid=grid(384), stream=stream0)
        buf286 = reinterpret_tensor(buf279, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf279  # reuse
        buf287 = buf276; del buf276  # reuse
        buf288 = buf275; del buf275  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42.run(buf283, buf25, buf286, buf287, buf288, 3072, 196, grid=grid(3072), stream=stream0)
        buf289 = buf278; del buf278  # reuse
        buf290 = buf277; del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf287, primals_169, buf288, buf289, buf290, 8, 384, grid=grid(8), stream=stream0)
        buf291 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf292 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf287, buf288, alias_174, alias_175, buf291, buf292, 384, 8, grid=grid(384), stream=stream0)
        buf293 = buf25; del buf25  # reuse
        buf294 = buf26; del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_44.run(buf293, buf282, buf283, buf286, alias_175, primals_169, buf290, alias_174, buf289, primals_168, buf294, 602112, grid=grid(602112), stream=stream0)
        del alias_174
        del alias_175
        del buf282
        del buf283
        del primals_168
        del primals_169
        buf295 = reinterpret_tensor(buf284, (1, 384, 1, 1, 13), (4992, 13, 4992, 4992, 1), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29.run(buf293, convolution_58, buf295, 4992, 121, grid=grid(4992), stream=stream0)
        del convolution_58
        buf296 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_30.run(buf295, buf296, 384, 13, grid=grid(384), stream=stream0)
        buf297 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf294, buf297, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf298 = aten.convolution_backward(buf294, clone_54, primals_335, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_54
        del primals_335
        buf299 = buf298[0]
        buf300 = buf298[1]
        del buf298
        buf301 = buf299; del buf299  # reuse
        # Source Nodes: [x_230], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf301, convolution_57, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del convolution_57
        buf302 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf301, buf302, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf303 = aten.convolution_backward(buf301, add_193, primals_333, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_193
        del buf301
        del primals_333
        buf304 = buf303[0]
        buf305 = buf303[1]
        del buf303
        buf306 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_34.run(buf304, buf24, avg_pool2d_27, add_190, primals_165, buf306, 6144, 98, grid=grid(6144), stream=stream0)
        buf307 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_35.run(buf306, buf307, 3072, 2, grid=grid(3072), stream=stream0)
        buf308 = buf287; del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_36.run(buf304, buf308, 3072, 196, grid=grid(3072), stream=stream0)
        buf309 = buf290; del buf290  # reuse
        buf310 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf307, primals_166, buf308, buf309, buf310, 8, 384, grid=grid(8), stream=stream0)
        buf311 = reinterpret_tensor(buf304, (8, 1, 384, 196), (75264, 602112, 196, 1), 0); del buf304  # reuse
        buf314 = buf293; del buf293  # reuse
        buf315 = buf294; del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_38.run(buf311, buf314, alias_177, primals_166, buf24, avg_pool2d_27, add_190, primals_165, buf310, alias_176, buf309, buf315, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_165
        del primals_166
        buf312 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf313 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf307, buf308, alias_176, alias_177, buf312, buf313, 384, 8, grid=grid(384), stream=stream0)
        del alias_176
        del alias_177
        buf316 = reinterpret_tensor(buf295, (1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), 0); del buf295  # reuse
        # Source Nodes: [sub_27], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_40.run(buf314, avg_pool2d_27, add_190, buf316, 4992, 121, grid=grid(4992), stream=stream0)
        del add_190
        del avg_pool2d_27
        buf317 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_27], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_41.run(buf316, buf317, 384, 13, grid=grid(384), stream=stream0)
        buf318 = reinterpret_tensor(buf311, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf311  # reuse
        buf319 = buf308; del buf308  # reuse
        buf320 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42.run(buf315, buf24, buf318, buf319, buf320, 3072, 196, grid=grid(3072), stream=stream0)
        buf321 = buf310; del buf310  # reuse
        buf322 = buf309; del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf319, primals_163, buf320, buf321, buf322, 8, 384, grid=grid(8), stream=stream0)
        buf323 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf324 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf319, buf320, alias_178, alias_179, buf323, buf324, 384, 8, grid=grid(384), stream=stream0)
        buf325 = buf24; del buf24  # reuse
        buf326 = buf286; del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_44.run(buf325, buf314, buf315, buf318, alias_179, primals_163, buf322, alias_178, buf321, primals_162, buf326, 602112, grid=grid(602112), stream=stream0)
        del alias_178
        del alias_179
        del buf314
        del buf315
        del primals_162
        del primals_163
        buf327 = reinterpret_tensor(buf316, (1, 384, 1, 1, 13), (4992, 13, 4992, 4992, 1), 0); del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29.run(buf325, convolution_56, buf327, 4992, 121, grid=grid(4992), stream=stream0)
        del convolution_56
        buf328 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_30.run(buf327, buf328, 384, 13, grid=grid(384), stream=stream0)
        buf329 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf326, buf329, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf330 = aten.convolution_backward(buf326, clone_52, primals_331, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_52
        del primals_331
        buf331 = buf330[0]
        buf332 = buf330[1]
        del buf330
        buf333 = buf331; del buf331  # reuse
        # Source Nodes: [x_222], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf333, convolution_55, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del convolution_55
        buf334 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf333, buf334, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf335 = aten.convolution_backward(buf333, add_186, primals_329, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_186
        del buf333
        del primals_329
        buf336 = buf335[0]
        buf337 = buf335[1]
        del buf335
        buf338 = buf306; del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_34.run(buf336, buf23, avg_pool2d_26, add_183, primals_159, buf338, 6144, 98, grid=grid(6144), stream=stream0)
        buf339 = buf320; del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_35.run(buf338, buf339, 3072, 2, grid=grid(3072), stream=stream0)
        buf340 = buf319; del buf319  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_36.run(buf336, buf340, 3072, 196, grid=grid(3072), stream=stream0)
        buf341 = buf322; del buf322  # reuse
        buf342 = buf321; del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf339, primals_160, buf340, buf341, buf342, 8, 384, grid=grid(8), stream=stream0)
        buf343 = reinterpret_tensor(buf336, (8, 1, 384, 196), (75264, 602112, 196, 1), 0); del buf336  # reuse
        buf346 = buf325; del buf325  # reuse
        buf347 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_38.run(buf343, buf346, alias_181, primals_160, buf23, avg_pool2d_26, add_183, primals_159, buf342, alias_180, buf341, buf347, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_159
        del primals_160
        buf344 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf345 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf339, buf340, alias_180, alias_181, buf344, buf345, 384, 8, grid=grid(384), stream=stream0)
        del alias_180
        del alias_181
        buf348 = reinterpret_tensor(buf327, (1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), 0); del buf327  # reuse
        # Source Nodes: [sub_26], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_40.run(buf346, avg_pool2d_26, add_183, buf348, 4992, 121, grid=grid(4992), stream=stream0)
        del add_183
        del avg_pool2d_26
        buf349 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_26], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_41.run(buf348, buf349, 384, 13, grid=grid(384), stream=stream0)
        buf350 = reinterpret_tensor(buf343, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf343  # reuse
        buf351 = buf340; del buf340  # reuse
        buf352 = buf339; del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42.run(buf347, buf23, buf350, buf351, buf352, 3072, 196, grid=grid(3072), stream=stream0)
        buf353 = buf342; del buf342  # reuse
        buf354 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf351, primals_157, buf352, buf353, buf354, 8, 384, grid=grid(8), stream=stream0)
        buf355 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf356 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf351, buf352, alias_182, alias_183, buf355, buf356, 384, 8, grid=grid(384), stream=stream0)
        buf357 = buf23; del buf23  # reuse
        buf358 = buf318; del buf318  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_44.run(buf357, buf346, buf347, buf350, alias_183, primals_157, buf354, alias_182, buf353, primals_156, buf358, 602112, grid=grid(602112), stream=stream0)
        del alias_182
        del alias_183
        del buf346
        del buf347
        del primals_156
        del primals_157
        buf359 = reinterpret_tensor(buf348, (1, 384, 1, 1, 13), (4992, 13, 4992, 4992, 1), 0); del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29.run(buf357, convolution_54, buf359, 4992, 121, grid=grid(4992), stream=stream0)
        del convolution_54
        buf360 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_30.run(buf359, buf360, 384, 13, grid=grid(384), stream=stream0)
        buf361 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf358, buf361, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf362 = aten.convolution_backward(buf358, clone_50, primals_327, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_50
        del primals_327
        buf363 = buf362[0]
        buf364 = buf362[1]
        del buf362
        buf365 = buf363; del buf363  # reuse
        # Source Nodes: [x_214], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf365, convolution_53, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del convolution_53
        buf366 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf365, buf366, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf367 = aten.convolution_backward(buf365, add_179, primals_325, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_179
        del buf365
        del primals_325
        buf368 = buf367[0]
        buf369 = buf367[1]
        del buf367
        buf370 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_34.run(buf368, buf22, avg_pool2d_25, add_176, primals_153, buf370, 6144, 98, grid=grid(6144), stream=stream0)
        buf371 = buf352; del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_35.run(buf370, buf371, 3072, 2, grid=grid(3072), stream=stream0)
        buf372 = buf351; del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_36.run(buf368, buf372, 3072, 196, grid=grid(3072), stream=stream0)
        buf373 = buf354; del buf354  # reuse
        buf374 = buf353; del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf371, primals_154, buf372, buf373, buf374, 8, 384, grid=grid(8), stream=stream0)
        buf375 = reinterpret_tensor(buf368, (8, 1, 384, 196), (75264, 602112, 196, 1), 0); del buf368  # reuse
        buf378 = buf357; del buf357  # reuse
        buf379 = buf358; del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_38.run(buf375, buf378, alias_185, primals_154, buf22, avg_pool2d_25, add_176, primals_153, buf374, alias_184, buf373, buf379, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_153
        del primals_154
        buf376 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf377 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf371, buf372, alias_184, alias_185, buf376, buf377, 384, 8, grid=grid(384), stream=stream0)
        del alias_184
        del alias_185
        buf380 = reinterpret_tensor(buf359, (1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), 0); del buf359  # reuse
        # Source Nodes: [sub_25], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_40.run(buf378, avg_pool2d_25, add_176, buf380, 4992, 121, grid=grid(4992), stream=stream0)
        del add_176
        del avg_pool2d_25
        buf381 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_25], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_41.run(buf380, buf381, 384, 13, grid=grid(384), stream=stream0)
        buf382 = reinterpret_tensor(buf375, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf375  # reuse
        buf383 = buf372; del buf372  # reuse
        buf384 = buf371; del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42.run(buf379, buf22, buf382, buf383, buf384, 3072, 196, grid=grid(3072), stream=stream0)
        buf385 = buf374; del buf374  # reuse
        buf386 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf383, primals_151, buf384, buf385, buf386, 8, 384, grid=grid(8), stream=stream0)
        buf387 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf388 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf383, buf384, alias_186, alias_187, buf387, buf388, 384, 8, grid=grid(384), stream=stream0)
        buf389 = buf22; del buf22  # reuse
        buf390 = buf350; del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_44.run(buf389, buf378, buf379, buf382, alias_187, primals_151, buf386, alias_186, buf385, primals_150, buf390, 602112, grid=grid(602112), stream=stream0)
        del alias_186
        del alias_187
        del buf378
        del buf379
        del primals_150
        del primals_151
        buf391 = reinterpret_tensor(buf380, (1, 384, 1, 1, 13), (4992, 13, 4992, 4992, 1), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29.run(buf389, convolution_52, buf391, 4992, 121, grid=grid(4992), stream=stream0)
        del convolution_52
        buf392 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_30.run(buf391, buf392, 384, 13, grid=grid(384), stream=stream0)
        buf393 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf390, buf393, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf394 = aten.convolution_backward(buf390, clone_48, primals_323, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_48
        del primals_323
        buf395 = buf394[0]
        buf396 = buf394[1]
        del buf394
        buf397 = buf395; del buf395  # reuse
        # Source Nodes: [x_206], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf397, convolution_51, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del convolution_51
        buf398 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf397, buf398, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf399 = aten.convolution_backward(buf397, add_172, primals_321, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_172
        del buf397
        del primals_321
        buf400 = buf399[0]
        buf401 = buf399[1]
        del buf399
        buf402 = buf370; del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_34.run(buf400, buf21, avg_pool2d_24, add_169, primals_147, buf402, 6144, 98, grid=grid(6144), stream=stream0)
        buf403 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_35.run(buf402, buf403, 3072, 2, grid=grid(3072), stream=stream0)
        buf404 = buf383; del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_36.run(buf400, buf404, 3072, 196, grid=grid(3072), stream=stream0)
        buf405 = buf386; del buf386  # reuse
        buf406 = buf385; del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf403, primals_148, buf404, buf405, buf406, 8, 384, grid=grid(8), stream=stream0)
        buf407 = reinterpret_tensor(buf400, (8, 1, 384, 196), (75264, 602112, 196, 1), 0); del buf400  # reuse
        buf410 = buf389; del buf389  # reuse
        buf411 = buf390; del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_38.run(buf407, buf410, alias_189, primals_148, buf21, avg_pool2d_24, add_169, primals_147, buf406, alias_188, buf405, buf411, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_147
        del primals_148
        buf408 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf409 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf403, buf404, alias_188, alias_189, buf408, buf409, 384, 8, grid=grid(384), stream=stream0)
        del alias_188
        del alias_189
        buf412 = reinterpret_tensor(buf391, (1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), 0); del buf391  # reuse
        # Source Nodes: [sub_24], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_40.run(buf410, avg_pool2d_24, add_169, buf412, 4992, 121, grid=grid(4992), stream=stream0)
        del add_169
        del avg_pool2d_24
        buf413 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_24], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_41.run(buf412, buf413, 384, 13, grid=grid(384), stream=stream0)
        buf414 = reinterpret_tensor(buf407, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf407  # reuse
        buf415 = buf404; del buf404  # reuse
        buf416 = buf403; del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42.run(buf411, buf21, buf414, buf415, buf416, 3072, 196, grid=grid(3072), stream=stream0)
        buf417 = buf406; del buf406  # reuse
        buf418 = buf405; del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf415, primals_145, buf416, buf417, buf418, 8, 384, grid=grid(8), stream=stream0)
        buf419 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf420 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf415, buf416, alias_190, alias_191, buf419, buf420, 384, 8, grid=grid(384), stream=stream0)
        buf421 = buf21; del buf21  # reuse
        buf422 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_44.run(buf421, buf410, buf411, buf414, alias_191, primals_145, buf418, alias_190, buf417, primals_144, buf422, 602112, grid=grid(602112), stream=stream0)
        del alias_190
        del alias_191
        del buf410
        del buf411
        del primals_144
        del primals_145
        buf423 = reinterpret_tensor(buf412, (1, 384, 1, 1, 13), (4992, 13, 4992, 4992, 1), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29.run(buf421, convolution_50, buf423, 4992, 121, grid=grid(4992), stream=stream0)
        del convolution_50
        buf424 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_30.run(buf423, buf424, 384, 13, grid=grid(384), stream=stream0)
        buf425 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf422, buf425, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf426 = aten.convolution_backward(buf422, clone_46, primals_319, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_46
        del primals_319
        buf427 = buf426[0]
        buf428 = buf426[1]
        del buf426
        buf429 = buf427; del buf427  # reuse
        # Source Nodes: [x_198], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf429, convolution_49, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del convolution_49
        buf430 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf429, buf430, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf431 = aten.convolution_backward(buf429, add_165, primals_317, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_165
        del buf429
        del primals_317
        buf432 = buf431[0]
        buf433 = buf431[1]
        del buf431
        buf434 = buf402; del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_34.run(buf432, buf20, avg_pool2d_23, add_162, primals_141, buf434, 6144, 98, grid=grid(6144), stream=stream0)
        buf435 = buf416; del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_35.run(buf434, buf435, 3072, 2, grid=grid(3072), stream=stream0)
        buf436 = buf415; del buf415  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_36.run(buf432, buf436, 3072, 196, grid=grid(3072), stream=stream0)
        buf437 = buf418; del buf418  # reuse
        buf438 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf435, primals_142, buf436, buf437, buf438, 8, 384, grid=grid(8), stream=stream0)
        buf439 = reinterpret_tensor(buf432, (8, 1, 384, 196), (75264, 602112, 196, 1), 0); del buf432  # reuse
        buf442 = buf421; del buf421  # reuse
        buf443 = buf422; del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_38.run(buf439, buf442, alias_193, primals_142, buf20, avg_pool2d_23, add_162, primals_141, buf438, alias_192, buf437, buf443, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_141
        del primals_142
        buf440 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf441 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf435, buf436, alias_192, alias_193, buf440, buf441, 384, 8, grid=grid(384), stream=stream0)
        del alias_192
        del alias_193
        buf444 = reinterpret_tensor(buf423, (1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), 0); del buf423  # reuse
        # Source Nodes: [sub_23], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_40.run(buf442, avg_pool2d_23, add_162, buf444, 4992, 121, grid=grid(4992), stream=stream0)
        del add_162
        del avg_pool2d_23
        buf445 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_23], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_41.run(buf444, buf445, 384, 13, grid=grid(384), stream=stream0)
        buf446 = reinterpret_tensor(buf439, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf439  # reuse
        buf447 = buf436; del buf436  # reuse
        buf448 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42.run(buf443, buf20, buf446, buf447, buf448, 3072, 196, grid=grid(3072), stream=stream0)
        buf449 = buf438; del buf438  # reuse
        buf450 = buf437; del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf447, primals_139, buf448, buf449, buf450, 8, 384, grid=grid(8), stream=stream0)
        buf451 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf452 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf447, buf448, alias_194, alias_195, buf451, buf452, 384, 8, grid=grid(384), stream=stream0)
        buf453 = buf20; del buf20  # reuse
        buf454 = buf414; del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_44.run(buf453, buf442, buf443, buf446, alias_195, primals_139, buf450, alias_194, buf449, primals_138, buf454, 602112, grid=grid(602112), stream=stream0)
        del alias_194
        del alias_195
        del buf442
        del buf443
        del primals_138
        del primals_139
        buf455 = reinterpret_tensor(buf444, (1, 384, 1, 1, 13), (4992, 13, 4992, 4992, 1), 0); del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29.run(buf453, convolution_48, buf455, 4992, 121, grid=grid(4992), stream=stream0)
        del convolution_48
        buf456 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_30.run(buf455, buf456, 384, 13, grid=grid(384), stream=stream0)
        buf457 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf454, buf457, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf458 = aten.convolution_backward(buf454, clone_44, primals_315, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_44
        del primals_315
        buf459 = buf458[0]
        buf460 = buf458[1]
        del buf458
        buf461 = buf459; del buf459  # reuse
        # Source Nodes: [x_190], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf461, convolution_47, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del convolution_47
        buf462 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf461, buf462, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf463 = aten.convolution_backward(buf461, add_158, primals_313, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_158
        del buf461
        del primals_313
        buf464 = buf463[0]
        buf465 = buf463[1]
        del buf463
        buf466 = buf434; del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_34.run(buf464, buf19, avg_pool2d_22, add_155, primals_135, buf466, 6144, 98, grid=grid(6144), stream=stream0)
        buf467 = buf448; del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_35.run(buf466, buf467, 3072, 2, grid=grid(3072), stream=stream0)
        buf468 = buf447; del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_36.run(buf464, buf468, 3072, 196, grid=grid(3072), stream=stream0)
        buf469 = buf450; del buf450  # reuse
        buf470 = buf449; del buf449  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf467, primals_136, buf468, buf469, buf470, 8, 384, grid=grid(8), stream=stream0)
        buf471 = reinterpret_tensor(buf464, (8, 1, 384, 196), (75264, 602112, 196, 1), 0); del buf464  # reuse
        buf474 = buf453; del buf453  # reuse
        buf475 = buf454; del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_38.run(buf471, buf474, alias_197, primals_136, buf19, avg_pool2d_22, add_155, primals_135, buf470, alias_196, buf469, buf475, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_135
        del primals_136
        buf472 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf473 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf467, buf468, alias_196, alias_197, buf472, buf473, 384, 8, grid=grid(384), stream=stream0)
        del alias_196
        del alias_197
        buf476 = reinterpret_tensor(buf455, (1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), 0); del buf455  # reuse
        # Source Nodes: [sub_22], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_40.run(buf474, avg_pool2d_22, add_155, buf476, 4992, 121, grid=grid(4992), stream=stream0)
        del add_155
        del avg_pool2d_22
        buf477 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_22], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_41.run(buf476, buf477, 384, 13, grid=grid(384), stream=stream0)
        buf478 = reinterpret_tensor(buf471, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf471  # reuse
        buf479 = buf468; del buf468  # reuse
        buf480 = buf467; del buf467  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42.run(buf475, buf19, buf478, buf479, buf480, 3072, 196, grid=grid(3072), stream=stream0)
        buf481 = buf470; del buf470  # reuse
        buf482 = buf469; del buf469  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf479, primals_133, buf480, buf481, buf482, 8, 384, grid=grid(8), stream=stream0)
        buf483 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf484 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf479, buf480, alias_198, alias_199, buf483, buf484, 384, 8, grid=grid(384), stream=stream0)
        buf485 = buf19; del buf19  # reuse
        buf486 = buf446; del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_44.run(buf485, buf474, buf475, buf478, alias_199, primals_133, buf482, alias_198, buf481, primals_132, buf486, 602112, grid=grid(602112), stream=stream0)
        del alias_198
        del alias_199
        del buf474
        del buf475
        del primals_132
        del primals_133
        buf487 = reinterpret_tensor(buf476, (1, 384, 1, 1, 13), (4992, 13, 4992, 4992, 1), 0); del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29.run(buf485, convolution_46, buf487, 4992, 121, grid=grid(4992), stream=stream0)
        del convolution_46
        buf488 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_30.run(buf487, buf488, 384, 13, grid=grid(384), stream=stream0)
        buf489 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf486, buf489, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf490 = aten.convolution_backward(buf486, clone_42, primals_311, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_42
        del primals_311
        buf491 = buf490[0]
        buf492 = buf490[1]
        del buf490
        buf493 = buf491; del buf491  # reuse
        # Source Nodes: [x_182], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf493, convolution_45, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del convolution_45
        buf494 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf493, buf494, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf495 = aten.convolution_backward(buf493, add_151, primals_309, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_151
        del buf493
        del primals_309
        buf496 = buf495[0]
        buf497 = buf495[1]
        del buf495
        buf498 = buf466; del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_34.run(buf496, buf18, avg_pool2d_21, add_148, primals_129, buf498, 6144, 98, grid=grid(6144), stream=stream0)
        buf499 = buf480; del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_35.run(buf498, buf499, 3072, 2, grid=grid(3072), stream=stream0)
        buf500 = buf479; del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_36.run(buf496, buf500, 3072, 196, grid=grid(3072), stream=stream0)
        buf501 = buf482; del buf482  # reuse
        buf502 = buf481; del buf481  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf499, primals_130, buf500, buf501, buf502, 8, 384, grid=grid(8), stream=stream0)
        buf503 = reinterpret_tensor(buf496, (8, 1, 384, 196), (75264, 602112, 196, 1), 0); del buf496  # reuse
        buf506 = buf485; del buf485  # reuse
        buf507 = buf486; del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_38.run(buf503, buf506, alias_201, primals_130, buf18, avg_pool2d_21, add_148, primals_129, buf502, alias_200, buf501, buf507, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_129
        del primals_130
        buf504 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf505 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf499, buf500, alias_200, alias_201, buf504, buf505, 384, 8, grid=grid(384), stream=stream0)
        del alias_200
        del alias_201
        buf508 = reinterpret_tensor(buf487, (1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), 0); del buf487  # reuse
        # Source Nodes: [sub_21], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_40.run(buf506, avg_pool2d_21, add_148, buf508, 4992, 121, grid=grid(4992), stream=stream0)
        del add_148
        del avg_pool2d_21
        buf509 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_21], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_41.run(buf508, buf509, 384, 13, grid=grid(384), stream=stream0)
        buf510 = reinterpret_tensor(buf503, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf503  # reuse
        buf511 = buf500; del buf500  # reuse
        buf512 = buf499; del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42.run(buf507, buf18, buf510, buf511, buf512, 3072, 196, grid=grid(3072), stream=stream0)
        buf513 = buf502; del buf502  # reuse
        buf514 = buf501; del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf511, primals_127, buf512, buf513, buf514, 8, 384, grid=grid(8), stream=stream0)
        buf515 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf516 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf511, buf512, alias_202, alias_203, buf515, buf516, 384, 8, grid=grid(384), stream=stream0)
        buf517 = buf18; del buf18  # reuse
        buf518 = buf478; del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_44.run(buf517, buf506, buf507, buf510, alias_203, primals_127, buf514, alias_202, buf513, primals_126, buf518, 602112, grid=grid(602112), stream=stream0)
        del alias_202
        del alias_203
        del buf506
        del buf507
        del primals_126
        del primals_127
        buf519 = reinterpret_tensor(buf508, (1, 384, 1, 1, 13), (4992, 13, 4992, 4992, 1), 0); del buf508  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29.run(buf517, convolution_44, buf519, 4992, 121, grid=grid(4992), stream=stream0)
        del convolution_44
        buf520 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_30.run(buf519, buf520, 384, 13, grid=grid(384), stream=stream0)
        buf521 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf518, buf521, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf522 = aten.convolution_backward(buf518, clone_40, primals_307, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_40
        del primals_307
        buf523 = buf522[0]
        buf524 = buf522[1]
        del buf522
        buf525 = buf523; del buf523  # reuse
        # Source Nodes: [x_174], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf525, convolution_43, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del convolution_43
        buf526 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf525, buf526, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf527 = aten.convolution_backward(buf525, add_144, primals_305, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_144
        del buf525
        del primals_305
        buf528 = buf527[0]
        buf529 = buf527[1]
        del buf527
        buf530 = buf498; del buf498  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_34.run(buf528, buf17, avg_pool2d_20, add_141, primals_123, buf530, 6144, 98, grid=grid(6144), stream=stream0)
        buf531 = buf512; del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_35.run(buf530, buf531, 3072, 2, grid=grid(3072), stream=stream0)
        buf532 = buf511; del buf511  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_36.run(buf528, buf532, 3072, 196, grid=grid(3072), stream=stream0)
        buf533 = buf514; del buf514  # reuse
        buf534 = buf513; del buf513  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf531, primals_124, buf532, buf533, buf534, 8, 384, grid=grid(8), stream=stream0)
        buf535 = reinterpret_tensor(buf528, (8, 1, 384, 196), (75264, 602112, 196, 1), 0); del buf528  # reuse
        buf538 = buf517; del buf517  # reuse
        buf539 = buf518; del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_38.run(buf535, buf538, alias_205, primals_124, buf17, avg_pool2d_20, add_141, primals_123, buf534, alias_204, buf533, buf539, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_123
        del primals_124
        buf536 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf537 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf531, buf532, alias_204, alias_205, buf536, buf537, 384, 8, grid=grid(384), stream=stream0)
        del alias_204
        del alias_205
        buf540 = reinterpret_tensor(buf519, (1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), 0); del buf519  # reuse
        # Source Nodes: [sub_20], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_40.run(buf538, avg_pool2d_20, add_141, buf540, 4992, 121, grid=grid(4992), stream=stream0)
        del add_141
        del avg_pool2d_20
        buf541 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_20], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_41.run(buf540, buf541, 384, 13, grid=grid(384), stream=stream0)
        buf542 = reinterpret_tensor(buf535, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf535  # reuse
        buf543 = buf532; del buf532  # reuse
        buf544 = buf531; del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42.run(buf539, buf17, buf542, buf543, buf544, 3072, 196, grid=grid(3072), stream=stream0)
        buf545 = buf534; del buf534  # reuse
        buf546 = buf533; del buf533  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf543, primals_121, buf544, buf545, buf546, 8, 384, grid=grid(8), stream=stream0)
        buf547 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf548 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf543, buf544, alias_206, alias_207, buf547, buf548, 384, 8, grid=grid(384), stream=stream0)
        buf549 = buf17; del buf17  # reuse
        buf550 = buf510; del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_44.run(buf549, buf538, buf539, buf542, alias_207, primals_121, buf546, alias_206, buf545, primals_120, buf550, 602112, grid=grid(602112), stream=stream0)
        del alias_206
        del alias_207
        del buf538
        del buf539
        del primals_120
        del primals_121
        buf551 = reinterpret_tensor(buf540, (1, 384, 1, 1, 13), (4992, 13, 4992, 4992, 1), 0); del buf540  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29.run(buf549, convolution_42, buf551, 4992, 121, grid=grid(4992), stream=stream0)
        del convolution_42
        buf552 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_30.run(buf551, buf552, 384, 13, grid=grid(384), stream=stream0)
        buf553 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf550, buf553, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf554 = aten.convolution_backward(buf550, clone_38, primals_303, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_38
        del primals_303
        buf555 = buf554[0]
        buf556 = buf554[1]
        del buf554
        buf557 = buf555; del buf555  # reuse
        # Source Nodes: [x_166], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf557, convolution_41, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del convolution_41
        buf558 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf557, buf558, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf559 = aten.convolution_backward(buf557, add_137, primals_301, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_137
        del buf557
        del primals_301
        buf560 = buf559[0]
        buf561 = buf559[1]
        del buf559
        buf562 = buf530; del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_34.run(buf560, buf16, avg_pool2d_19, add_134, primals_117, buf562, 6144, 98, grid=grid(6144), stream=stream0)
        buf563 = buf544; del buf544  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_35.run(buf562, buf563, 3072, 2, grid=grid(3072), stream=stream0)
        buf564 = buf543; del buf543  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_36.run(buf560, buf564, 3072, 196, grid=grid(3072), stream=stream0)
        buf565 = buf546; del buf546  # reuse
        buf566 = buf545; del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf563, primals_118, buf564, buf565, buf566, 8, 384, grid=grid(8), stream=stream0)
        buf567 = reinterpret_tensor(buf560, (8, 1, 384, 196), (75264, 602112, 196, 1), 0); del buf560  # reuse
        buf570 = buf549; del buf549  # reuse
        buf571 = buf550; del buf550  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_38.run(buf567, buf570, alias_209, primals_118, buf16, avg_pool2d_19, add_134, primals_117, buf566, alias_208, buf565, buf571, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_117
        del primals_118
        buf568 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf569 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf563, buf564, alias_208, alias_209, buf568, buf569, 384, 8, grid=grid(384), stream=stream0)
        del alias_208
        del alias_209
        buf572 = reinterpret_tensor(buf551, (1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), 0); del buf551  # reuse
        # Source Nodes: [sub_19], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_40.run(buf570, avg_pool2d_19, add_134, buf572, 4992, 121, grid=grid(4992), stream=stream0)
        del add_134
        del avg_pool2d_19
        buf573 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_19], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_41.run(buf572, buf573, 384, 13, grid=grid(384), stream=stream0)
        buf574 = reinterpret_tensor(buf567, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf567  # reuse
        buf575 = buf564; del buf564  # reuse
        buf576 = buf563; del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42.run(buf571, buf16, buf574, buf575, buf576, 3072, 196, grid=grid(3072), stream=stream0)
        buf577 = buf566; del buf566  # reuse
        buf578 = buf565; del buf565  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf575, primals_115, buf576, buf577, buf578, 8, 384, grid=grid(8), stream=stream0)
        buf579 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf580 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf575, buf576, alias_210, alias_211, buf579, buf580, 384, 8, grid=grid(384), stream=stream0)
        buf581 = buf16; del buf16  # reuse
        buf582 = buf542; del buf542  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_44.run(buf581, buf570, buf571, buf574, alias_211, primals_115, buf578, alias_210, buf577, primals_114, buf582, 602112, grid=grid(602112), stream=stream0)
        del alias_210
        del alias_211
        del buf570
        del buf571
        del primals_114
        del primals_115
        buf583 = reinterpret_tensor(buf572, (1, 384, 1, 1, 13), (4992, 13, 4992, 4992, 1), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29.run(buf581, convolution_40, buf583, 4992, 121, grid=grid(4992), stream=stream0)
        del convolution_40
        buf584 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_30.run(buf583, buf584, 384, 13, grid=grid(384), stream=stream0)
        buf585 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf582, buf585, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf586 = aten.convolution_backward(buf582, clone_36, primals_299, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_36
        del primals_299
        buf587 = buf586[0]
        buf588 = buf586[1]
        del buf586
        buf589 = buf587; del buf587  # reuse
        # Source Nodes: [x_158], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf589, convolution_39, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del convolution_39
        buf590 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf589, buf590, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf591 = aten.convolution_backward(buf589, add_130, primals_297, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_130
        del buf589
        del primals_297
        buf592 = buf591[0]
        buf593 = buf591[1]
        del buf591
        buf594 = buf562; del buf562  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_34.run(buf592, buf15, avg_pool2d_18, add_127, primals_111, buf594, 6144, 98, grid=grid(6144), stream=stream0)
        buf595 = buf576; del buf576  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_35.run(buf594, buf595, 3072, 2, grid=grid(3072), stream=stream0)
        buf596 = buf575; del buf575  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_36.run(buf592, buf596, 3072, 196, grid=grid(3072), stream=stream0)
        buf597 = buf578; del buf578  # reuse
        buf598 = buf577; del buf577  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf595, primals_112, buf596, buf597, buf598, 8, 384, grid=grid(8), stream=stream0)
        buf599 = reinterpret_tensor(buf592, (8, 1, 384, 196), (75264, 602112, 196, 1), 0); del buf592  # reuse
        buf602 = buf581; del buf581  # reuse
        buf603 = buf582; del buf582  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_38.run(buf599, buf602, alias_213, primals_112, buf15, avg_pool2d_18, add_127, primals_111, buf598, alias_212, buf597, buf603, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_111
        del primals_112
        buf600 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf601 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf595, buf596, alias_212, alias_213, buf600, buf601, 384, 8, grid=grid(384), stream=stream0)
        del alias_212
        del alias_213
        buf604 = reinterpret_tensor(buf583, (1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), 0); del buf583  # reuse
        # Source Nodes: [sub_18], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_40.run(buf602, avg_pool2d_18, add_127, buf604, 4992, 121, grid=grid(4992), stream=stream0)
        del add_127
        del avg_pool2d_18
        buf605 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_18], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_41.run(buf604, buf605, 384, 13, grid=grid(384), stream=stream0)
        buf606 = reinterpret_tensor(buf599, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf599  # reuse
        buf607 = buf596; del buf596  # reuse
        buf608 = buf595; del buf595  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42.run(buf603, buf15, buf606, buf607, buf608, 3072, 196, grid=grid(3072), stream=stream0)
        buf609 = buf598; del buf598  # reuse
        buf610 = buf597; del buf597  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf607, primals_109, buf608, buf609, buf610, 8, 384, grid=grid(8), stream=stream0)
        buf611 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf612 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf607, buf608, alias_214, alias_215, buf611, buf612, 384, 8, grid=grid(384), stream=stream0)
        buf613 = buf15; del buf15  # reuse
        buf614 = buf574; del buf574  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_44.run(buf613, buf602, buf603, buf606, alias_215, primals_109, buf610, alias_214, buf609, primals_108, buf614, 602112, grid=grid(602112), stream=stream0)
        del alias_214
        del alias_215
        del buf602
        del buf603
        del primals_108
        del primals_109
        buf615 = reinterpret_tensor(buf604, (1, 384, 1, 1, 13), (4992, 13, 4992, 4992, 1), 0); del buf604  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29.run(buf613, convolution_38, buf615, 4992, 121, grid=grid(4992), stream=stream0)
        del convolution_38
        buf616 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_30.run(buf615, buf616, 384, 13, grid=grid(384), stream=stream0)
        buf617 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf614, buf617, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf618 = aten.convolution_backward(buf614, clone_34, primals_295, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_34
        del primals_295
        buf619 = buf618[0]
        buf620 = buf618[1]
        del buf618
        buf621 = buf619; del buf619  # reuse
        # Source Nodes: [x_150], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf621, convolution_37, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del convolution_37
        buf622 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf621, buf622, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf623 = aten.convolution_backward(buf621, add_123, primals_293, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_123
        del buf621
        del primals_293
        buf624 = buf623[0]
        buf625 = buf623[1]
        del buf623
        buf626 = buf594; del buf594  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_34.run(buf624, buf14, avg_pool2d_17, add_120, primals_105, buf626, 6144, 98, grid=grid(6144), stream=stream0)
        buf627 = buf608; del buf608  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_35.run(buf626, buf627, 3072, 2, grid=grid(3072), stream=stream0)
        buf628 = buf607; del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_36.run(buf624, buf628, 3072, 196, grid=grid(3072), stream=stream0)
        buf629 = buf610; del buf610  # reuse
        buf630 = buf609; del buf609  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf627, primals_106, buf628, buf629, buf630, 8, 384, grid=grid(8), stream=stream0)
        buf631 = reinterpret_tensor(buf624, (8, 1, 384, 196), (75264, 602112, 196, 1), 0); del buf624  # reuse
        buf634 = buf613; del buf613  # reuse
        buf635 = buf614; del buf614  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_38.run(buf631, buf634, alias_217, primals_106, buf14, avg_pool2d_17, add_120, primals_105, buf630, alias_216, buf629, buf635, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_105
        del primals_106
        buf632 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf633 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf627, buf628, alias_216, alias_217, buf632, buf633, 384, 8, grid=grid(384), stream=stream0)
        del alias_216
        del alias_217
        buf636 = reinterpret_tensor(buf615, (1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), 0); del buf615  # reuse
        # Source Nodes: [sub_17], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_40.run(buf634, avg_pool2d_17, add_120, buf636, 4992, 121, grid=grid(4992), stream=stream0)
        del add_120
        del avg_pool2d_17
        buf637 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_17], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_41.run(buf636, buf637, 384, 13, grid=grid(384), stream=stream0)
        buf638 = reinterpret_tensor(buf631, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf631  # reuse
        buf639 = buf628; del buf628  # reuse
        buf640 = buf627; del buf627  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42.run(buf635, buf14, buf638, buf639, buf640, 3072, 196, grid=grid(3072), stream=stream0)
        buf641 = buf630; del buf630  # reuse
        buf642 = buf629; del buf629  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf639, primals_103, buf640, buf641, buf642, 8, 384, grid=grid(8), stream=stream0)
        buf643 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf644 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf639, buf640, alias_218, alias_219, buf643, buf644, 384, 8, grid=grid(384), stream=stream0)
        buf645 = buf14; del buf14  # reuse
        buf646 = buf606; del buf606  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_44.run(buf645, buf634, buf635, buf638, alias_219, primals_103, buf642, alias_218, buf641, primals_102, buf646, 602112, grid=grid(602112), stream=stream0)
        del alias_218
        del alias_219
        del buf634
        del buf635
        del primals_102
        del primals_103
        buf647 = reinterpret_tensor(buf636, (1, 384, 1, 1, 13), (4992, 13, 4992, 4992, 1), 0); del buf636  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29.run(buf645, convolution_36, buf647, 4992, 121, grid=grid(4992), stream=stream0)
        del convolution_36
        buf648 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_30.run(buf647, buf648, 384, 13, grid=grid(384), stream=stream0)
        buf649 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf646, buf649, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf650 = aten.convolution_backward(buf646, clone_32, primals_291, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_32
        del primals_291
        buf651 = buf650[0]
        buf652 = buf650[1]
        del buf650
        buf653 = buf651; del buf651  # reuse
        # Source Nodes: [x_142], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf653, convolution_35, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del convolution_35
        buf654 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf653, buf654, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf655 = aten.convolution_backward(buf653, add_116, primals_289, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_116
        del buf653
        del primals_289
        buf656 = buf655[0]
        buf657 = buf655[1]
        del buf655
        buf658 = buf626; del buf626  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_34.run(buf656, buf13, avg_pool2d_16, add_113, primals_99, buf658, 6144, 98, grid=grid(6144), stream=stream0)
        buf659 = buf640; del buf640  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_35.run(buf658, buf659, 3072, 2, grid=grid(3072), stream=stream0)
        buf660 = buf639; del buf639  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_36.run(buf656, buf660, 3072, 196, grid=grid(3072), stream=stream0)
        buf661 = buf642; del buf642  # reuse
        buf662 = buf641; del buf641  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf659, primals_100, buf660, buf661, buf662, 8, 384, grid=grid(8), stream=stream0)
        buf663 = reinterpret_tensor(buf656, (8, 1, 384, 196), (75264, 602112, 196, 1), 0); del buf656  # reuse
        buf666 = buf645; del buf645  # reuse
        buf667 = buf646; del buf646  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_38.run(buf663, buf666, alias_221, primals_100, buf13, avg_pool2d_16, add_113, primals_99, buf662, alias_220, buf661, buf667, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_100
        del primals_99
        buf664 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf665 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf659, buf660, alias_220, alias_221, buf664, buf665, 384, 8, grid=grid(384), stream=stream0)
        del alias_220
        del alias_221
        buf668 = reinterpret_tensor(buf647, (1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), 0); del buf647  # reuse
        # Source Nodes: [sub_16], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_40.run(buf666, avg_pool2d_16, add_113, buf668, 4992, 121, grid=grid(4992), stream=stream0)
        del add_113
        del avg_pool2d_16
        buf669 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_16], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_41.run(buf668, buf669, 384, 13, grid=grid(384), stream=stream0)
        buf670 = reinterpret_tensor(buf663, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf663  # reuse
        buf671 = buf660; del buf660  # reuse
        buf672 = buf659; del buf659  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42.run(buf667, buf13, buf670, buf671, buf672, 3072, 196, grid=grid(3072), stream=stream0)
        buf673 = buf662; del buf662  # reuse
        buf674 = buf661; del buf661  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf671, primals_97, buf672, buf673, buf674, 8, 384, grid=grid(8), stream=stream0)
        buf675 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf676 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf671, buf672, alias_222, alias_223, buf675, buf676, 384, 8, grid=grid(384), stream=stream0)
        buf677 = buf13; del buf13  # reuse
        buf678 = buf638; del buf638  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_44.run(buf677, buf666, buf667, buf670, alias_223, primals_97, buf674, alias_222, buf673, primals_96, buf678, 602112, grid=grid(602112), stream=stream0)
        del alias_222
        del alias_223
        del buf666
        del buf667
        del primals_96
        del primals_97
        buf679 = reinterpret_tensor(buf668, (1, 384, 1, 1, 13), (4992, 13, 4992, 4992, 1), 0); del buf668  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29.run(buf677, convolution_34, buf679, 4992, 121, grid=grid(4992), stream=stream0)
        del convolution_34
        buf680 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_30.run(buf679, buf680, 384, 13, grid=grid(384), stream=stream0)
        buf681 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf678, buf681, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf682 = aten.convolution_backward(buf678, clone_30, primals_287, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_30
        del primals_287
        buf683 = buf682[0]
        buf684 = buf682[1]
        del buf682
        buf685 = buf683; del buf683  # reuse
        # Source Nodes: [x_134], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf685, convolution_33, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del convolution_33
        buf686 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf685, buf686, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf687 = aten.convolution_backward(buf685, add_109, primals_285, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_109
        del buf685
        del primals_285
        buf688 = buf687[0]
        buf689 = buf687[1]
        del buf687
        buf690 = buf658; del buf658  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_34.run(buf688, buf12, avg_pool2d_15, add_106, primals_93, buf690, 6144, 98, grid=grid(6144), stream=stream0)
        buf691 = buf672; del buf672  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_35.run(buf690, buf691, 3072, 2, grid=grid(3072), stream=stream0)
        buf692 = buf671; del buf671  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_36.run(buf688, buf692, 3072, 196, grid=grid(3072), stream=stream0)
        buf693 = buf674; del buf674  # reuse
        buf694 = buf673; del buf673  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf691, primals_94, buf692, buf693, buf694, 8, 384, grid=grid(8), stream=stream0)
        buf695 = reinterpret_tensor(buf688, (8, 1, 384, 196), (75264, 602112, 196, 1), 0); del buf688  # reuse
        buf698 = buf677; del buf677  # reuse
        buf699 = buf678; del buf678  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_38.run(buf695, buf698, alias_225, primals_94, buf12, avg_pool2d_15, add_106, primals_93, buf694, alias_224, buf693, buf699, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_93
        del primals_94
        buf696 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf697 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf691, buf692, alias_224, alias_225, buf696, buf697, 384, 8, grid=grid(384), stream=stream0)
        del alias_224
        del alias_225
        buf700 = reinterpret_tensor(buf679, (1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), 0); del buf679  # reuse
        # Source Nodes: [sub_15], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_40.run(buf698, avg_pool2d_15, add_106, buf700, 4992, 121, grid=grid(4992), stream=stream0)
        del add_106
        del avg_pool2d_15
        buf701 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_15], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_41.run(buf700, buf701, 384, 13, grid=grid(384), stream=stream0)
        buf702 = reinterpret_tensor(buf695, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf695  # reuse
        buf703 = buf692; del buf692  # reuse
        buf704 = buf691; del buf691  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42.run(buf699, buf12, buf702, buf703, buf704, 3072, 196, grid=grid(3072), stream=stream0)
        buf705 = buf694; del buf694  # reuse
        buf706 = buf693; del buf693  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf703, primals_91, buf704, buf705, buf706, 8, 384, grid=grid(8), stream=stream0)
        buf707 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf708 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf703, buf704, alias_226, alias_227, buf707, buf708, 384, 8, grid=grid(384), stream=stream0)
        buf709 = buf12; del buf12  # reuse
        buf710 = buf670; del buf670  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_44.run(buf709, buf698, buf699, buf702, alias_227, primals_91, buf706, alias_226, buf705, primals_90, buf710, 602112, grid=grid(602112), stream=stream0)
        del alias_226
        del alias_227
        del buf698
        del buf699
        del primals_90
        del primals_91
        buf711 = reinterpret_tensor(buf700, (1, 384, 1, 1, 13), (4992, 13, 4992, 4992, 1), 0); del buf700  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29.run(buf709, convolution_32, buf711, 4992, 121, grid=grid(4992), stream=stream0)
        del convolution_32
        buf712 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_30.run(buf711, buf712, 384, 13, grid=grid(384), stream=stream0)
        buf713 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf710, buf713, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf714 = aten.convolution_backward(buf710, clone_28, primals_283, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_28
        del primals_283
        buf715 = buf714[0]
        buf716 = buf714[1]
        del buf714
        buf717 = buf715; del buf715  # reuse
        # Source Nodes: [x_126], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf717, convolution_31, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del convolution_31
        buf718 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf717, buf718, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf719 = aten.convolution_backward(buf717, add_102, primals_281, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_102
        del buf717
        del primals_281
        buf720 = buf719[0]
        buf721 = buf719[1]
        del buf719
        buf722 = buf690; del buf690  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_34.run(buf720, buf11, avg_pool2d_14, add_99, primals_87, buf722, 6144, 98, grid=grid(6144), stream=stream0)
        buf723 = buf704; del buf704  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_35.run(buf722, buf723, 3072, 2, grid=grid(3072), stream=stream0)
        buf724 = buf703; del buf703  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_36.run(buf720, buf724, 3072, 196, grid=grid(3072), stream=stream0)
        buf725 = buf706; del buf706  # reuse
        buf726 = buf705; del buf705  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf723, primals_88, buf724, buf725, buf726, 8, 384, grid=grid(8), stream=stream0)
        buf727 = reinterpret_tensor(buf720, (8, 1, 384, 196), (75264, 602112, 196, 1), 0); del buf720  # reuse
        buf730 = buf709; del buf709  # reuse
        buf731 = buf710; del buf710  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_38.run(buf727, buf730, alias_229, primals_88, buf11, avg_pool2d_14, add_99, primals_87, buf726, alias_228, buf725, buf731, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_87
        del primals_88
        buf728 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf729 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf723, buf724, alias_228, alias_229, buf728, buf729, 384, 8, grid=grid(384), stream=stream0)
        del alias_228
        del alias_229
        buf732 = reinterpret_tensor(buf711, (1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), 0); del buf711  # reuse
        # Source Nodes: [sub_14], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_40.run(buf730, avg_pool2d_14, add_99, buf732, 4992, 121, grid=grid(4992), stream=stream0)
        del add_99
        del avg_pool2d_14
        buf733 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_14], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_41.run(buf732, buf733, 384, 13, grid=grid(384), stream=stream0)
        buf734 = reinterpret_tensor(buf727, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf727  # reuse
        buf735 = buf724; del buf724  # reuse
        buf736 = buf723; del buf723  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42.run(buf731, buf11, buf734, buf735, buf736, 3072, 196, grid=grid(3072), stream=stream0)
        buf737 = buf726; del buf726  # reuse
        buf738 = buf725; del buf725  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf735, primals_85, buf736, buf737, buf738, 8, 384, grid=grid(8), stream=stream0)
        buf739 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf740 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf735, buf736, alias_230, alias_231, buf739, buf740, 384, 8, grid=grid(384), stream=stream0)
        buf741 = buf11; del buf11  # reuse
        buf742 = buf702; del buf702  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_44.run(buf741, buf730, buf731, buf734, alias_231, primals_85, buf738, alias_230, buf737, primals_84, buf742, 602112, grid=grid(602112), stream=stream0)
        del alias_230
        del alias_231
        del buf730
        del buf731
        del primals_84
        del primals_85
        buf743 = reinterpret_tensor(buf732, (1, 384, 1, 1, 13), (4992, 13, 4992, 4992, 1), 0); del buf732  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29.run(buf741, convolution_30, buf743, 4992, 121, grid=grid(4992), stream=stream0)
        del convolution_30
        buf744 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_30.run(buf743, buf744, 384, 13, grid=grid(384), stream=stream0)
        buf745 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf742, buf745, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf746 = aten.convolution_backward(buf742, clone_26, primals_279, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_26
        del primals_279
        buf747 = buf746[0]
        buf748 = buf746[1]
        del buf746
        buf749 = buf747; del buf747  # reuse
        # Source Nodes: [x_118], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf749, convolution_29, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del convolution_29
        buf750 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf749, buf750, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf751 = aten.convolution_backward(buf749, add_95, primals_277, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_95
        del primals_277
        buf752 = buf751[0]
        buf753 = buf751[1]
        del buf751
        buf754 = buf722; del buf722  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_34.run(buf752, buf10, avg_pool2d_13, add_92, primals_81, buf754, 6144, 98, grid=grid(6144), stream=stream0)
        buf755 = buf736; del buf736  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_35.run(buf754, buf755, 3072, 2, grid=grid(3072), stream=stream0)
        buf756 = buf735; del buf735  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_36.run(buf752, buf756, 3072, 196, grid=grid(3072), stream=stream0)
        buf757 = buf738; del buf738  # reuse
        buf758 = buf737; del buf737  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf755, primals_82, buf756, buf757, buf758, 8, 384, grid=grid(8), stream=stream0)
        buf759 = reinterpret_tensor(buf752, (8, 1, 384, 196), (75264, 602112, 196, 1), 0); del buf752  # reuse
        buf762 = buf741; del buf741  # reuse
        buf763 = buf742; del buf742  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_38.run(buf759, buf762, alias_233, primals_82, buf10, avg_pool2d_13, add_92, primals_81, buf758, alias_232, buf757, buf763, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_81
        del primals_82
        buf760 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf761 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf755, buf756, alias_232, alias_233, buf760, buf761, 384, 8, grid=grid(384), stream=stream0)
        del alias_232
        del alias_233
        buf764 = reinterpret_tensor(buf743, (1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), 0); del buf743  # reuse
        # Source Nodes: [sub_13], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_40.run(buf762, avg_pool2d_13, add_92, buf764, 4992, 121, grid=grid(4992), stream=stream0)
        del add_92
        del avg_pool2d_13
        buf765 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_13], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_41.run(buf764, buf765, 384, 13, grid=grid(384), stream=stream0)
        buf766 = reinterpret_tensor(buf759, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf759  # reuse
        buf767 = buf756; del buf756  # reuse
        buf768 = buf755; del buf755  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_42.run(buf763, buf10, buf766, buf767, buf768, 3072, 196, grid=grid(3072), stream=stream0)
        buf769 = buf758; del buf758  # reuse
        buf770 = buf757; del buf757  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf767, primals_79, buf768, buf769, buf770, 8, 384, grid=grid(8), stream=stream0)
        buf771 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf772 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf767, buf768, alias_234, alias_235, buf771, buf772, 384, 8, grid=grid(384), stream=stream0)
        buf773 = buf10; del buf10  # reuse
        buf774 = buf734; del buf734  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_44.run(buf773, buf762, buf763, buf766, alias_235, primals_79, buf770, alias_234, buf769, primals_78, buf774, 602112, grid=grid(602112), stream=stream0)
        del alias_234
        del alias_235
        del buf762
        del buf763
        del buf766
        del primals_78
        del primals_79
        buf775 = reinterpret_tensor(buf764, (1, 384, 1, 1, 13), (4992, 13, 4992, 4992, 1), 0); del buf764  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_29.run(buf773, convolution_28, buf775, 4992, 121, grid=grid(4992), stream=stream0)
        del convolution_28
        buf776 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_30.run(buf775, buf776, 384, 13, grid=grid(384), stream=stream0)
        buf777 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf774, buf777, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf778 = aten.convolution_backward(buf774, clone_24, primals_275, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_24
        del primals_275
        buf779 = buf778[0]
        buf780 = buf778[1]
        del buf778
        buf781 = buf779; del buf779  # reuse
        # Source Nodes: [x_110], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_32.run(buf781, convolution_27, 12288, 196, grid=grid(12288, 196), stream=stream0)
        del convolution_27
        buf782 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf781, buf782, 1536, 1568, grid=grid(1536), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf783 = aten.convolution_backward(buf781, add_88, primals_273, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_88
        del primals_273
        buf784 = buf783[0]
        buf785 = buf783[1]
        del buf783
        buf786 = reinterpret_tensor(buf754, (8, 384, 2), (768, 1, 384), 0); del buf754  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_45.run(buf784, convolution_26, avg_pool2d_12, add_85, primals_75, buf786, 6144, 98, grid=grid(6144), stream=stream0)
        buf787 = buf768; del buf768  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_46.run(buf786, buf787, 3072, 2, grid=grid(3072), stream=stream0)
        del buf786
        buf788 = buf767; del buf767  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_36.run(buf784, buf788, 3072, 196, grid=grid(3072), stream=stream0)
        buf789 = buf770; del buf770  # reuse
        buf790 = buf769; del buf769  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf787, primals_76, buf788, buf789, buf790, 8, 384, grid=grid(8), stream=stream0)
        buf791 = reinterpret_tensor(buf774, (8, 1, 384, 196), (75264, 602112, 1, 384), 0); del buf774  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_poi_fused_native_group_norm_backward_47.run(buf784, alias_237, primals_76, convolution_26, avg_pool2d_12, add_85, primals_75, buf790, alias_236, buf789, buf791, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del primals_76
        buf792 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf793 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf787, buf788, alias_236, alias_237, buf792, buf793, 384, 8, grid=grid(384), stream=stream0)
        buf794 = buf773; del buf773  # reuse
        buf795 = buf784; del buf784  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_48.run(buf794, buf791, buf790, alias_236, buf789, alias_237, primals_75, buf795, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del alias_236
        del alias_237
        del primals_75
        buf796 = reinterpret_tensor(buf775, (1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), 0); del buf775  # reuse
        # Source Nodes: [sub_12], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_40.run(buf794, avg_pool2d_12, add_85, buf796, 4992, 121, grid=grid(4992), stream=stream0)
        del add_85
        del avg_pool2d_12
        buf797 = empty((1, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_12], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_41.run(buf796, buf797, 384, 13, grid=grid(384), stream=stream0)
        del buf796
        buf798 = reinterpret_tensor(buf791, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf791  # reuse
        buf799 = buf788; del buf788  # reuse
        buf800 = buf787; del buf787  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_49.run(buf795, convolution_26, buf798, buf799, buf800, 3072, 196, grid=grid(3072), stream=stream0)
        buf801 = buf790; del buf790  # reuse
        buf802 = buf789; del buf789  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_37.run(buf799, primals_73, buf800, buf801, buf802, 8, 384, grid=grid(8), stream=stream0)
        buf803 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf804 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_39.run(buf799, buf800, alias_238, alias_239, buf803, buf804, 384, 8, grid=grid(384), stream=stream0)
        del buf799
        del buf800
        buf805 = buf794; del buf794  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_50.run(buf805, buf795, buf798, alias_239, primals_73, convolution_26, buf802, alias_238, buf801, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del alias_238
        del alias_239
        del buf795
        del buf798
        del convolution_26
        del primals_73
        buf806 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf805, buf806, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf807 = aten.convolution_backward(buf805, add_83, primals_271, [384], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_83
        del buf805
        del primals_271
        buf808 = buf807[0]
        buf809 = buf807[1]
        del buf807
        buf810 = reinterpret_tensor(buf202, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_51.run(buf808, primals_72, buf810, 1204224, grid=grid(1204224), stream=stream0)
        del primals_72
        buf811 = empty_strided((1, 192, 1, 1, 49), (9408, 49, 9408, 9408, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_52.run(buf808, convolution_25, buf811, 9408, 128, grid=grid(9408), stream=stream0)
        del convolution_25
        buf812 = empty((1, 192, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_53.run(buf811, buf812, 192, 49, grid=grid(192), stream=stream0)
        buf813 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_54.run(buf810, buf813, 192, 6272, grid=grid(192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf814 = aten.convolution_backward(buf810, clone_22, primals_269, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_22
        del primals_269
        buf815 = buf814[0]
        buf816 = buf814[1]
        del buf814
        buf817 = buf815; del buf815  # reuse
        # Source Nodes: [x_98], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_55.run(buf817, convolution_24, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_24
        buf818 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_56.run(buf817, buf818, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf819 = aten.convolution_backward(buf817, add_81, primals_267, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_81
        del buf817
        del primals_267
        buf820 = buf819[0]
        buf821 = buf819[1]
        del buf819
        buf822 = empty((8, 192, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_57.run(buf820, buf9, avg_pool2d_11, add_78, primals_69, buf822, 10752, 112, grid=grid(10752), stream=stream0)
        buf823 = empty((8, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_58.run(buf822, buf823, 1536, 7, grid=grid(1536), stream=stream0)
        buf824 = empty((8, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_59.run(buf820, buf824, 1536, 784, grid=grid(1536), stream=stream0)
        buf825 = buf802; del buf802  # reuse
        buf826 = buf801; del buf801  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_60.run(buf823, primals_70, buf824, buf825, buf826, 8, 192, grid=grid(8), stream=stream0)
        buf827 = reinterpret_tensor(buf820, (8, 1, 192, 784), (150528, 1204224, 784, 1), 0); del buf820  # reuse
        buf830 = buf808; del buf808  # reuse
        buf831 = buf810; del buf810  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_61.run(buf827, buf830, alias_241, primals_70, buf9, avg_pool2d_11, add_78, primals_69, buf826, alias_240, buf825, buf831, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_69
        del primals_70
        buf828 = empty((1, 192), device='cuda', dtype=torch.float32)
        buf829 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_62.run(buf823, buf824, alias_240, alias_241, buf828, buf829, 192, 8, grid=grid(192), stream=stream0)
        del alias_240
        del alias_241
        buf832 = reinterpret_tensor(buf811, (1, 192, 1, 1, 49), (9408, 1, 9408, 9408, 192), 0); del buf811  # reuse
        # Source Nodes: [sub_11], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_63.run(buf830, avg_pool2d_11, add_78, buf832, 9408, 128, grid=grid(9408), stream=stream0)
        del add_78
        del avg_pool2d_11
        buf833 = empty((1, 192, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_11], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_64.run(buf832, buf833, 192, 49, grid=grid(192), stream=stream0)
        buf834 = reinterpret_tensor(buf827, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf827  # reuse
        buf835 = buf824; del buf824  # reuse
        buf836 = buf823; del buf823  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_65.run(buf831, buf9, buf834, buf835, buf836, 1536, 784, grid=grid(1536), stream=stream0)
        buf837 = buf826; del buf826  # reuse
        buf838 = buf825; del buf825  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_60.run(buf835, primals_67, buf836, buf837, buf838, 8, 192, grid=grid(8), stream=stream0)
        buf839 = empty((1, 192), device='cuda', dtype=torch.float32)
        buf840 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_62.run(buf835, buf836, alias_242, alias_243, buf839, buf840, 192, 8, grid=grid(192), stream=stream0)
        buf841 = buf830; del buf830  # reuse
        buf842 = reinterpret_tensor(buf171, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_66.run(buf841, buf831, buf834, alias_243, primals_67, buf9, buf838, alias_242, buf837, primals_66, buf842, 1204224, grid=grid(1204224), stream=stream0)
        del alias_242
        del alias_243
        del buf831
        del buf834
        del primals_66
        del primals_67
        buf843 = reinterpret_tensor(buf832, (1, 192, 1, 1, 49), (9408, 49, 9408, 9408, 1), 0); del buf832  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_52.run(buf841, convolution_23, buf843, 9408, 128, grid=grid(9408), stream=stream0)
        del convolution_23
        buf844 = empty((1, 192, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_53.run(buf843, buf844, 192, 49, grid=grid(192), stream=stream0)
        buf845 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_54.run(buf842, buf845, 192, 6272, grid=grid(192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf846 = aten.convolution_backward(buf842, clone_20, primals_265, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_20
        del primals_265
        buf847 = buf846[0]
        buf848 = buf846[1]
        del buf846
        buf849 = buf847; del buf847  # reuse
        # Source Nodes: [x_90], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_55.run(buf849, convolution_22, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_22
        buf850 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_56.run(buf849, buf850, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf851 = aten.convolution_backward(buf849, add_74, primals_263, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_74
        del buf849
        del primals_263
        buf852 = buf851[0]
        buf853 = buf851[1]
        del buf851
        buf854 = buf822; del buf822  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_57.run(buf852, buf8, avg_pool2d_10, add_71, primals_63, buf854, 10752, 112, grid=grid(10752), stream=stream0)
        buf855 = buf836; del buf836  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_58.run(buf854, buf855, 1536, 7, grid=grid(1536), stream=stream0)
        buf856 = buf835; del buf835  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_59.run(buf852, buf856, 1536, 784, grid=grid(1536), stream=stream0)
        buf857 = buf838; del buf838  # reuse
        buf858 = buf837; del buf837  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_60.run(buf855, primals_64, buf856, buf857, buf858, 8, 192, grid=grid(8), stream=stream0)
        buf859 = reinterpret_tensor(buf852, (8, 1, 192, 784), (150528, 1204224, 784, 1), 0); del buf852  # reuse
        buf862 = buf841; del buf841  # reuse
        buf863 = buf842; del buf842  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_61.run(buf859, buf862, alias_245, primals_64, buf8, avg_pool2d_10, add_71, primals_63, buf858, alias_244, buf857, buf863, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_63
        del primals_64
        buf860 = empty((1, 192), device='cuda', dtype=torch.float32)
        buf861 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_62.run(buf855, buf856, alias_244, alias_245, buf860, buf861, 192, 8, grid=grid(192), stream=stream0)
        del alias_244
        del alias_245
        buf864 = reinterpret_tensor(buf843, (1, 192, 1, 1, 49), (9408, 1, 9408, 9408, 192), 0); del buf843  # reuse
        # Source Nodes: [sub_10], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_63.run(buf862, avg_pool2d_10, add_71, buf864, 9408, 128, grid=grid(9408), stream=stream0)
        del add_71
        del avg_pool2d_10
        buf865 = empty((1, 192, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_10], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_64.run(buf864, buf865, 192, 49, grid=grid(192), stream=stream0)
        buf866 = reinterpret_tensor(buf859, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf859  # reuse
        buf867 = buf856; del buf856  # reuse
        buf868 = buf855; del buf855  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_65.run(buf863, buf8, buf866, buf867, buf868, 1536, 784, grid=grid(1536), stream=stream0)
        buf869 = buf858; del buf858  # reuse
        buf870 = buf857; del buf857  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_60.run(buf867, primals_61, buf868, buf869, buf870, 8, 192, grid=grid(8), stream=stream0)
        buf871 = empty((1, 192), device='cuda', dtype=torch.float32)
        buf872 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_62.run(buf867, buf868, alias_246, alias_247, buf871, buf872, 192, 8, grid=grid(192), stream=stream0)
        buf873 = buf8; del buf8  # reuse
        buf874 = buf9; del buf9  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_67.run(buf873, buf862, buf863, buf866, alias_247, primals_61, buf870, alias_246, buf869, primals_60, buf874, 1204224, grid=grid(1204224), stream=stream0)
        del alias_246
        del alias_247
        del buf862
        del buf863
        del primals_60
        del primals_61
        buf875 = reinterpret_tensor(buf864, (1, 192, 1, 1, 49), (9408, 49, 9408, 9408, 1), 0); del buf864  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_52.run(buf873, convolution_21, buf875, 9408, 128, grid=grid(9408), stream=stream0)
        del convolution_21
        buf876 = empty((1, 192, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_53.run(buf875, buf876, 192, 49, grid=grid(192), stream=stream0)
        buf877 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_54.run(buf874, buf877, 192, 6272, grid=grid(192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf878 = aten.convolution_backward(buf874, clone_18, primals_261, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_18
        del primals_261
        buf879 = buf878[0]
        buf880 = buf878[1]
        del buf878
        buf881 = buf879; del buf879  # reuse
        # Source Nodes: [x_82], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_55.run(buf881, convolution_20, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_20
        buf882 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_56.run(buf881, buf882, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf883 = aten.convolution_backward(buf881, add_67, primals_259, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_67
        del buf881
        del primals_259
        buf884 = buf883[0]
        buf885 = buf883[1]
        del buf883
        buf886 = buf854; del buf854  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_57.run(buf884, buf7, avg_pool2d_9, add_64, primals_57, buf886, 10752, 112, grid=grid(10752), stream=stream0)
        buf887 = buf868; del buf868  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_58.run(buf886, buf887, 1536, 7, grid=grid(1536), stream=stream0)
        buf888 = buf867; del buf867  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_59.run(buf884, buf888, 1536, 784, grid=grid(1536), stream=stream0)
        buf889 = buf870; del buf870  # reuse
        buf890 = buf869; del buf869  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_60.run(buf887, primals_58, buf888, buf889, buf890, 8, 192, grid=grid(8), stream=stream0)
        buf891 = reinterpret_tensor(buf884, (8, 1, 192, 784), (150528, 1204224, 784, 1), 0); del buf884  # reuse
        buf894 = buf873; del buf873  # reuse
        buf895 = buf874; del buf874  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_61.run(buf891, buf894, alias_249, primals_58, buf7, avg_pool2d_9, add_64, primals_57, buf890, alias_248, buf889, buf895, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_57
        del primals_58
        buf892 = empty((1, 192), device='cuda', dtype=torch.float32)
        buf893 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_62.run(buf887, buf888, alias_248, alias_249, buf892, buf893, 192, 8, grid=grid(192), stream=stream0)
        del alias_248
        del alias_249
        buf896 = reinterpret_tensor(buf875, (1, 192, 1, 1, 49), (9408, 1, 9408, 9408, 192), 0); del buf875  # reuse
        # Source Nodes: [sub_9], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_63.run(buf894, avg_pool2d_9, add_64, buf896, 9408, 128, grid=grid(9408), stream=stream0)
        del add_64
        del avg_pool2d_9
        buf897 = empty((1, 192, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_9], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_64.run(buf896, buf897, 192, 49, grid=grid(192), stream=stream0)
        buf898 = reinterpret_tensor(buf891, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf891  # reuse
        buf899 = buf888; del buf888  # reuse
        buf900 = buf887; del buf887  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_65.run(buf895, buf7, buf898, buf899, buf900, 1536, 784, grid=grid(1536), stream=stream0)
        buf901 = buf890; del buf890  # reuse
        buf902 = buf889; del buf889  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_60.run(buf899, primals_55, buf900, buf901, buf902, 8, 192, grid=grid(8), stream=stream0)
        buf903 = empty((1, 192), device='cuda', dtype=torch.float32)
        buf904 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_62.run(buf899, buf900, alias_250, alias_251, buf903, buf904, 192, 8, grid=grid(192), stream=stream0)
        buf905 = buf7; del buf7  # reuse
        buf906 = buf866; del buf866  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_67.run(buf905, buf894, buf895, buf898, alias_251, primals_55, buf902, alias_250, buf901, primals_54, buf906, 1204224, grid=grid(1204224), stream=stream0)
        del alias_250
        del alias_251
        del buf894
        del buf895
        del primals_54
        del primals_55
        buf907 = reinterpret_tensor(buf896, (1, 192, 1, 1, 49), (9408, 49, 9408, 9408, 1), 0); del buf896  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_52.run(buf905, convolution_19, buf907, 9408, 128, grid=grid(9408), stream=stream0)
        del convolution_19
        buf908 = empty((1, 192, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_53.run(buf907, buf908, 192, 49, grid=grid(192), stream=stream0)
        buf909 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_54.run(buf906, buf909, 192, 6272, grid=grid(192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf910 = aten.convolution_backward(buf906, clone_16, primals_257, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_16
        del primals_257
        buf911 = buf910[0]
        buf912 = buf910[1]
        del buf910
        buf913 = buf911; del buf911  # reuse
        # Source Nodes: [x_74], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_55.run(buf913, convolution_18, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_18
        buf914 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_56.run(buf913, buf914, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf915 = aten.convolution_backward(buf913, add_60, primals_255, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_60
        del buf913
        del primals_255
        buf916 = buf915[0]
        buf917 = buf915[1]
        del buf915
        buf918 = buf886; del buf886  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_57.run(buf916, buf6, avg_pool2d_8, add_57, primals_51, buf918, 10752, 112, grid=grid(10752), stream=stream0)
        buf919 = buf900; del buf900  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_58.run(buf918, buf919, 1536, 7, grid=grid(1536), stream=stream0)
        buf920 = buf899; del buf899  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_59.run(buf916, buf920, 1536, 784, grid=grid(1536), stream=stream0)
        buf921 = buf902; del buf902  # reuse
        buf922 = buf901; del buf901  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_60.run(buf919, primals_52, buf920, buf921, buf922, 8, 192, grid=grid(8), stream=stream0)
        buf923 = reinterpret_tensor(buf916, (8, 1, 192, 784), (150528, 1204224, 784, 1), 0); del buf916  # reuse
        buf926 = buf905; del buf905  # reuse
        buf927 = buf906; del buf906  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_61.run(buf923, buf926, alias_253, primals_52, buf6, avg_pool2d_8, add_57, primals_51, buf922, alias_252, buf921, buf927, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_51
        del primals_52
        buf924 = empty((1, 192), device='cuda', dtype=torch.float32)
        buf925 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_62.run(buf919, buf920, alias_252, alias_253, buf924, buf925, 192, 8, grid=grid(192), stream=stream0)
        del alias_252
        del alias_253
        buf928 = reinterpret_tensor(buf907, (1, 192, 1, 1, 49), (9408, 1, 9408, 9408, 192), 0); del buf907  # reuse
        # Source Nodes: [sub_8], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_63.run(buf926, avg_pool2d_8, add_57, buf928, 9408, 128, grid=grid(9408), stream=stream0)
        del add_57
        del avg_pool2d_8
        buf929 = empty((1, 192, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_8], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_64.run(buf928, buf929, 192, 49, grid=grid(192), stream=stream0)
        buf930 = reinterpret_tensor(buf923, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf923  # reuse
        buf931 = buf920; del buf920  # reuse
        buf932 = buf919; del buf919  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_65.run(buf927, buf6, buf930, buf931, buf932, 1536, 784, grid=grid(1536), stream=stream0)
        buf933 = buf922; del buf922  # reuse
        buf934 = buf921; del buf921  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_60.run(buf931, primals_49, buf932, buf933, buf934, 8, 192, grid=grid(8), stream=stream0)
        buf935 = empty((1, 192), device='cuda', dtype=torch.float32)
        buf936 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_62.run(buf931, buf932, alias_254, alias_255, buf935, buf936, 192, 8, grid=grid(192), stream=stream0)
        buf937 = buf6; del buf6  # reuse
        buf938 = buf898; del buf898  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_67.run(buf937, buf926, buf927, buf930, alias_255, primals_49, buf934, alias_254, buf933, primals_48, buf938, 1204224, grid=grid(1204224), stream=stream0)
        del alias_254
        del alias_255
        del buf926
        del buf927
        del primals_48
        del primals_49
        buf939 = reinterpret_tensor(buf928, (1, 192, 1, 1, 49), (9408, 49, 9408, 9408, 1), 0); del buf928  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_52.run(buf937, convolution_17, buf939, 9408, 128, grid=grid(9408), stream=stream0)
        del convolution_17
        buf940 = empty((1, 192, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_53.run(buf939, buf940, 192, 49, grid=grid(192), stream=stream0)
        buf941 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_54.run(buf938, buf941, 192, 6272, grid=grid(192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf942 = aten.convolution_backward(buf938, clone_14, primals_253, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_14
        del primals_253
        buf943 = buf942[0]
        buf944 = buf942[1]
        del buf942
        buf945 = buf943; del buf943  # reuse
        # Source Nodes: [x_66], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_55.run(buf945, convolution_16, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_16
        buf946 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_56.run(buf945, buf946, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf947 = aten.convolution_backward(buf945, add_53, primals_251, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_53
        del buf945
        del primals_251
        buf948 = buf947[0]
        buf949 = buf947[1]
        del buf947
        buf950 = buf918; del buf918  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_57.run(buf948, buf5, avg_pool2d_7, add_50, primals_45, buf950, 10752, 112, grid=grid(10752), stream=stream0)
        buf951 = buf932; del buf932  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_58.run(buf950, buf951, 1536, 7, grid=grid(1536), stream=stream0)
        buf952 = buf931; del buf931  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_59.run(buf948, buf952, 1536, 784, grid=grid(1536), stream=stream0)
        buf953 = buf934; del buf934  # reuse
        buf954 = buf933; del buf933  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_60.run(buf951, primals_46, buf952, buf953, buf954, 8, 192, grid=grid(8), stream=stream0)
        buf955 = reinterpret_tensor(buf948, (8, 1, 192, 784), (150528, 1204224, 784, 1), 0); del buf948  # reuse
        buf958 = buf937; del buf937  # reuse
        buf959 = buf938; del buf938  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_61.run(buf955, buf958, alias_257, primals_46, buf5, avg_pool2d_7, add_50, primals_45, buf954, alias_256, buf953, buf959, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_45
        del primals_46
        buf956 = empty((1, 192), device='cuda', dtype=torch.float32)
        buf957 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_62.run(buf951, buf952, alias_256, alias_257, buf956, buf957, 192, 8, grid=grid(192), stream=stream0)
        del alias_256
        del alias_257
        buf960 = reinterpret_tensor(buf939, (1, 192, 1, 1, 49), (9408, 1, 9408, 9408, 192), 0); del buf939  # reuse
        # Source Nodes: [sub_7], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_63.run(buf958, avg_pool2d_7, add_50, buf960, 9408, 128, grid=grid(9408), stream=stream0)
        del add_50
        del avg_pool2d_7
        buf961 = empty((1, 192, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_7], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_64.run(buf960, buf961, 192, 49, grid=grid(192), stream=stream0)
        buf962 = reinterpret_tensor(buf955, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf955  # reuse
        buf963 = buf952; del buf952  # reuse
        buf964 = buf951; del buf951  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_65.run(buf959, buf5, buf962, buf963, buf964, 1536, 784, grid=grid(1536), stream=stream0)
        buf965 = buf954; del buf954  # reuse
        buf966 = buf953; del buf953  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_60.run(buf963, primals_43, buf964, buf965, buf966, 8, 192, grid=grid(8), stream=stream0)
        buf967 = empty((1, 192), device='cuda', dtype=torch.float32)
        buf968 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_62.run(buf963, buf964, alias_258, alias_259, buf967, buf968, 192, 8, grid=grid(192), stream=stream0)
        buf969 = buf5; del buf5  # reuse
        buf970 = buf930; del buf930  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_67.run(buf969, buf958, buf959, buf962, alias_259, primals_43, buf966, alias_258, buf965, primals_42, buf970, 1204224, grid=grid(1204224), stream=stream0)
        del alias_258
        del alias_259
        del buf958
        del buf959
        del buf962
        del primals_42
        del primals_43
        buf971 = reinterpret_tensor(buf960, (1, 192, 1, 1, 49), (9408, 49, 9408, 9408, 1), 0); del buf960  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_52.run(buf969, convolution_15, buf971, 9408, 128, grid=grid(9408), stream=stream0)
        del convolution_15
        buf972 = empty((1, 192, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_53.run(buf971, buf972, 192, 49, grid=grid(192), stream=stream0)
        buf973 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_54.run(buf970, buf973, 192, 6272, grid=grid(192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf974 = aten.convolution_backward(buf970, clone_12, primals_249, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_12
        del primals_249
        buf975 = buf974[0]
        buf976 = buf974[1]
        del buf974
        buf977 = buf975; del buf975  # reuse
        # Source Nodes: [x_58], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_55.run(buf977, convolution_14, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_14
        buf978 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_56.run(buf977, buf978, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf979 = aten.convolution_backward(buf977, add_46, primals_247, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_46
        del buf977
        del primals_247
        buf980 = buf979[0]
        buf981 = buf979[1]
        del buf979
        buf982 = reinterpret_tensor(buf950, (8, 192, 7), (1344, 1, 192), 0); del buf950  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_68.run(buf980, convolution_13, avg_pool2d_6, add_43, primals_39, buf982, 10752, 112, grid=grid(10752), stream=stream0)
        buf983 = buf964; del buf964  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_69.run(buf982, buf983, 1536, 7, grid=grid(1536), stream=stream0)
        del buf982
        buf984 = buf963; del buf963  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_59.run(buf980, buf984, 1536, 784, grid=grid(1536), stream=stream0)
        buf985 = buf966; del buf966  # reuse
        buf986 = buf965; del buf965  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_60.run(buf983, primals_40, buf984, buf985, buf986, 8, 192, grid=grid(8), stream=stream0)
        buf987 = reinterpret_tensor(buf970, (8, 1, 192, 784), (150528, 1204224, 1, 192), 0); del buf970  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_poi_fused_native_group_norm_backward_70.run(buf980, alias_261, primals_40, convolution_13, avg_pool2d_6, add_43, primals_39, buf986, alias_260, buf985, buf987, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del primals_40
        buf988 = empty((1, 192), device='cuda', dtype=torch.float32)
        buf989 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_62.run(buf983, buf984, alias_260, alias_261, buf988, buf989, 192, 8, grid=grid(192), stream=stream0)
        buf990 = buf969; del buf969  # reuse
        buf991 = buf980; del buf980  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_71.run(buf990, buf987, buf986, alias_260, buf985, alias_261, primals_39, buf991, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del alias_260
        del alias_261
        del primals_39
        buf992 = reinterpret_tensor(buf971, (1, 192, 1, 1, 49), (9408, 1, 9408, 9408, 192), 0); del buf971  # reuse
        # Source Nodes: [sub_6], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_63.run(buf990, avg_pool2d_6, add_43, buf992, 9408, 128, grid=grid(9408), stream=stream0)
        del add_43
        del avg_pool2d_6
        buf993 = empty((1, 192, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_6], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_per_fused_mul_sub_sum_64.run(buf992, buf993, 192, 49, grid=grid(192), stream=stream0)
        del buf992
        buf994 = reinterpret_tensor(buf987, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf987  # reuse
        buf995 = buf984; del buf984  # reuse
        buf996 = buf983; del buf983  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_per_fused_avg_pool2d_backward_native_group_norm_backward_72.run(buf991, convolution_13, buf994, buf995, buf996, 1536, 784, grid=grid(1536), stream=stream0)
        buf997 = buf986; del buf986  # reuse
        buf998 = buf985; del buf985  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_60.run(buf995, primals_37, buf996, buf997, buf998, 8, 192, grid=grid(8), stream=stream0)
        buf999 = empty((1, 192), device='cuda', dtype=torch.float32)
        buf1000 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_62.run(buf995, buf996, alias_262, alias_263, buf999, buf1000, 192, 8, grid=grid(192), stream=stream0)
        del buf995
        del buf996
        buf1001 = buf990; del buf990  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_73.run(buf1001, buf991, buf994, alias_263, primals_37, convolution_13, buf998, alias_262, buf997, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del alias_262
        del alias_263
        del buf991
        del buf994
        del convolution_13
        del primals_37
        buf1002 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_54.run(buf1001, buf1002, 192, 6272, grid=grid(192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1003 = aten.convolution_backward(buf1001, add_41, primals_245, [192], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_41
        del buf1001
        del primals_245
        buf1004 = buf1003[0]
        buf1005 = buf1003[1]
        del buf1003
        buf1006 = reinterpret_tensor(buf781, (8, 96, 56, 56), (301056, 3136, 56, 1), 0); del buf781  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_74.run(buf1004, primals_36, buf1006, 2408448, grid=grid(2408448), stream=stream0)
        del primals_36
        buf1007 = empty_strided((1, 96, 1, 1, 196), (18816, 196, 18816, 18816, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_75.run(buf1004, convolution_12, buf1007, 18816, 128, grid=grid(18816), stream=stream0)
        del convolution_12
        buf1008 = empty((1, 96, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_76.run(buf1007, buf1008, 96, 196, grid=grid(96), stream=stream0)
        buf1009 = empty_strided((96, 4), (1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_77.run(buf1006, buf1009, 384, 6272, grid=grid(384), stream=stream0)
        buf1010 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_78.run(buf1009, buf1010, 96, 4, grid=grid(96), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1011 = aten.convolution_backward(buf1006, clone_10, primals_243, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_10
        del primals_243
        buf1012 = buf1011[0]
        buf1013 = buf1011[1]
        del buf1011
        buf1014 = buf1012; del buf1012  # reuse
        # Source Nodes: [x_46], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_79.run(buf1014, convolution_11, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del convolution_11
        buf1015 = reinterpret_tensor(buf1009, (384, ), (1, ), 0); del buf1009  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_80.run(buf1014, buf1015, 384, 25088, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1016 = aten.convolution_backward(buf1014, add_39, primals_241, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_39
        del buf1014
        del primals_241
        buf1017 = buf1016[0]
        buf1018 = buf1016[1]
        del buf1016
        buf1019 = empty((8, 96, 25), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_81.run(buf1017, buf4, avg_pool2d_5, add_36, primals_33, buf1019, 19200, 126, grid=grid(19200), stream=stream0)
        buf1020 = empty((8, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_82.run(buf1019, buf1020, 768, 25, grid=grid(768), stream=stream0)
        buf1021 = empty((8, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_83.run(buf1017, buf1021, 768, 3136, grid=grid(768), stream=stream0)
        buf1022 = buf998; del buf998  # reuse
        buf1023 = buf997; del buf997  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_84.run(buf1020, primals_34, buf1021, buf1022, buf1023, 8, 96, grid=grid(8), stream=stream0)
        buf1024 = reinterpret_tensor(buf1017, (8, 1, 96, 3136), (301056, 2408448, 3136, 1), 0); del buf1017  # reuse
        buf1027 = buf1004; del buf1004  # reuse
        buf1028 = buf1006; del buf1006  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_85.run(buf1024, buf1027, alias_265, primals_34, buf4, avg_pool2d_5, add_36, primals_33, buf1023, alias_264, buf1022, buf1028, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_33
        del primals_34
        buf1025 = empty((1, 96), device='cuda', dtype=torch.float32)
        buf1026 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_86.run(buf1020, buf1021, alias_264, alias_265, buf1025, buf1026, 96, 8, grid=grid(96), stream=stream0)
        del alias_264
        del alias_265
        buf1029 = reinterpret_tensor(buf1007, (1, 96, 1, 1, 196), (18816, 1, 18816, 18816, 96), 0); del buf1007  # reuse
        # Source Nodes: [sub_5], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_87.run(buf1027, avg_pool2d_5, add_36, buf1029, 18816, 128, grid=grid(18816), stream=stream0)
        del add_36
        del avg_pool2d_5
        buf1030 = empty((1, 96, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_5], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_88.run(buf1029, buf1030, 96, 196, grid=grid(96), stream=stream0)
        buf1031 = reinterpret_tensor(buf1024, (8, 96, 56, 56), (301056, 3136, 56, 1), 0); del buf1024  # reuse
        buf1032 = buf1021; del buf1021  # reuse
        buf1033 = buf1020; del buf1020  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_red_fused_avg_pool2d_backward_native_group_norm_backward_89.run(buf1028, buf4, buf1031, buf1032, buf1033, 768, 3136, grid=grid(768), stream=stream0)
        buf1034 = buf1023; del buf1023  # reuse
        buf1035 = buf1022; del buf1022  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_84.run(buf1032, primals_31, buf1033, buf1034, buf1035, 8, 96, grid=grid(8), stream=stream0)
        buf1036 = empty((1, 96), device='cuda', dtype=torch.float32)
        buf1037 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_86.run(buf1032, buf1033, alias_266, alias_267, buf1036, buf1037, 96, 8, grid=grid(96), stream=stream0)
        buf1038 = buf1027; del buf1027  # reuse
        buf1039 = reinterpret_tensor(buf749, (8, 96, 56, 56), (301056, 3136, 56, 1), 0); del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_90.run(buf1038, buf1028, buf1031, alias_267, primals_31, buf4, buf1035, alias_266, buf1034, primals_30, buf1039, 2408448, grid=grid(2408448), stream=stream0)
        del alias_266
        del alias_267
        del buf1028
        del buf1031
        del primals_30
        del primals_31
        buf1040 = reinterpret_tensor(buf1029, (1, 96, 1, 1, 196), (18816, 196, 18816, 18816, 1), 0); del buf1029  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_75.run(buf1038, convolution_10, buf1040, 18816, 128, grid=grid(18816), stream=stream0)
        del convolution_10
        buf1041 = empty((1, 96, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_76.run(buf1040, buf1041, 96, 196, grid=grid(96), stream=stream0)
        buf1042 = empty_strided((96, 4), (1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_77.run(buf1039, buf1042, 384, 6272, grid=grid(384), stream=stream0)
        buf1043 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_78.run(buf1042, buf1043, 96, 4, grid=grid(96), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1044 = aten.convolution_backward(buf1039, clone_8, primals_239, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_8
        del primals_239
        buf1045 = buf1044[0]
        buf1046 = buf1044[1]
        del buf1044
        buf1047 = buf1045; del buf1045  # reuse
        # Source Nodes: [x_38], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_79.run(buf1047, convolution_9, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del convolution_9
        buf1048 = reinterpret_tensor(buf1042, (384, ), (1, ), 0); del buf1042  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_80.run(buf1047, buf1048, 384, 25088, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1049 = aten.convolution_backward(buf1047, add_32, primals_237, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_32
        del buf1047
        del primals_237
        buf1050 = buf1049[0]
        buf1051 = buf1049[1]
        del buf1049
        buf1052 = buf1019; del buf1019  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_81.run(buf1050, buf3, avg_pool2d_4, add_29, primals_27, buf1052, 19200, 126, grid=grid(19200), stream=stream0)
        buf1053 = buf1033; del buf1033  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_82.run(buf1052, buf1053, 768, 25, grid=grid(768), stream=stream0)
        buf1054 = buf1032; del buf1032  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_83.run(buf1050, buf1054, 768, 3136, grid=grid(768), stream=stream0)
        buf1055 = buf1035; del buf1035  # reuse
        buf1056 = buf1034; del buf1034  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_84.run(buf1053, primals_28, buf1054, buf1055, buf1056, 8, 96, grid=grid(8), stream=stream0)
        buf1057 = reinterpret_tensor(buf1050, (8, 1, 96, 3136), (301056, 2408448, 3136, 1), 0); del buf1050  # reuse
        buf1060 = buf1038; del buf1038  # reuse
        buf1061 = buf1039; del buf1039  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_85.run(buf1057, buf1060, alias_269, primals_28, buf3, avg_pool2d_4, add_29, primals_27, buf1056, alias_268, buf1055, buf1061, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_27
        del primals_28
        buf1058 = empty((1, 96), device='cuda', dtype=torch.float32)
        buf1059 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_86.run(buf1053, buf1054, alias_268, alias_269, buf1058, buf1059, 96, 8, grid=grid(96), stream=stream0)
        del alias_268
        del alias_269
        buf1062 = reinterpret_tensor(buf1040, (1, 96, 1, 1, 196), (18816, 1, 18816, 18816, 96), 0); del buf1040  # reuse
        # Source Nodes: [sub_4], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_87.run(buf1060, avg_pool2d_4, add_29, buf1062, 18816, 128, grid=grid(18816), stream=stream0)
        del add_29
        del avg_pool2d_4
        buf1063 = empty((1, 96, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_4], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_88.run(buf1062, buf1063, 96, 196, grid=grid(96), stream=stream0)
        buf1064 = reinterpret_tensor(buf1057, (8, 96, 56, 56), (301056, 3136, 56, 1), 0); del buf1057  # reuse
        buf1065 = buf1054; del buf1054  # reuse
        buf1066 = buf1053; del buf1053  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_red_fused_avg_pool2d_backward_native_group_norm_backward_89.run(buf1061, buf3, buf1064, buf1065, buf1066, 768, 3136, grid=grid(768), stream=stream0)
        buf1067 = buf1056; del buf1056  # reuse
        buf1068 = buf1055; del buf1055  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_84.run(buf1065, primals_25, buf1066, buf1067, buf1068, 8, 96, grid=grid(8), stream=stream0)
        buf1069 = empty((1, 96), device='cuda', dtype=torch.float32)
        buf1070 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_86.run(buf1065, buf1066, alias_270, alias_271, buf1069, buf1070, 96, 8, grid=grid(96), stream=stream0)
        buf1071 = buf1060; del buf1060  # reuse
        buf1072 = buf4; del buf4  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_90.run(buf1071, buf1061, buf1064, alias_271, primals_25, buf3, buf1068, alias_270, buf1067, primals_24, buf1072, 2408448, grid=grid(2408448), stream=stream0)
        del alias_270
        del alias_271
        del buf1061
        del buf1064
        del primals_24
        del primals_25
        buf1073 = reinterpret_tensor(buf1062, (1, 96, 1, 1, 196), (18816, 196, 18816, 18816, 1), 0); del buf1062  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_75.run(buf1071, convolution_8, buf1073, 18816, 128, grid=grid(18816), stream=stream0)
        del convolution_8
        buf1074 = empty((1, 96, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_76.run(buf1073, buf1074, 96, 196, grid=grid(96), stream=stream0)
        buf1075 = empty_strided((96, 4), (1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_77.run(buf1072, buf1075, 384, 6272, grid=grid(384), stream=stream0)
        buf1076 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_78.run(buf1075, buf1076, 96, 4, grid=grid(96), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1077 = aten.convolution_backward(buf1072, clone_6, primals_235, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_6
        del primals_235
        buf1078 = buf1077[0]
        buf1079 = buf1077[1]
        del buf1077
        buf1080 = buf1078; del buf1078  # reuse
        # Source Nodes: [x_30], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_79.run(buf1080, convolution_7, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del convolution_7
        buf1081 = reinterpret_tensor(buf1075, (384, ), (1, ), 0); del buf1075  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_80.run(buf1080, buf1081, 384, 25088, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1082 = aten.convolution_backward(buf1080, add_25, primals_233, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_25
        del buf1080
        del primals_233
        buf1083 = buf1082[0]
        buf1084 = buf1082[1]
        del buf1082
        buf1085 = buf1052; del buf1052  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_81.run(buf1083, buf2, avg_pool2d_3, add_22, primals_21, buf1085, 19200, 126, grid=grid(19200), stream=stream0)
        buf1086 = buf1066; del buf1066  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_82.run(buf1085, buf1086, 768, 25, grid=grid(768), stream=stream0)
        buf1087 = buf1065; del buf1065  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_83.run(buf1083, buf1087, 768, 3136, grid=grid(768), stream=stream0)
        buf1088 = buf1068; del buf1068  # reuse
        buf1089 = buf1067; del buf1067  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_84.run(buf1086, primals_22, buf1087, buf1088, buf1089, 8, 96, grid=grid(8), stream=stream0)
        buf1090 = reinterpret_tensor(buf1083, (8, 1, 96, 3136), (301056, 2408448, 3136, 1), 0); del buf1083  # reuse
        buf1093 = buf1071; del buf1071  # reuse
        buf1094 = buf1072; del buf1072  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_85.run(buf1090, buf1093, alias_273, primals_22, buf2, avg_pool2d_3, add_22, primals_21, buf1089, alias_272, buf1088, buf1094, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_21
        del primals_22
        buf1091 = empty((1, 96), device='cuda', dtype=torch.float32)
        buf1092 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_86.run(buf1086, buf1087, alias_272, alias_273, buf1091, buf1092, 96, 8, grid=grid(96), stream=stream0)
        del alias_272
        del alias_273
        buf1095 = reinterpret_tensor(buf1073, (1, 96, 1, 1, 196), (18816, 1, 18816, 18816, 96), 0); del buf1073  # reuse
        # Source Nodes: [sub_3], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_87.run(buf1093, avg_pool2d_3, add_22, buf1095, 18816, 128, grid=grid(18816), stream=stream0)
        del add_22
        del avg_pool2d_3
        buf1096 = empty((1, 96, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_3], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_88.run(buf1095, buf1096, 96, 196, grid=grid(96), stream=stream0)
        buf1097 = reinterpret_tensor(buf1090, (8, 96, 56, 56), (301056, 3136, 56, 1), 0); del buf1090  # reuse
        buf1098 = buf1087; del buf1087  # reuse
        buf1099 = buf1086; del buf1086  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_red_fused_avg_pool2d_backward_native_group_norm_backward_89.run(buf1094, buf2, buf1097, buf1098, buf1099, 768, 3136, grid=grid(768), stream=stream0)
        buf1100 = buf1089; del buf1089  # reuse
        buf1101 = buf1088; del buf1088  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_84.run(buf1098, primals_19, buf1099, buf1100, buf1101, 8, 96, grid=grid(8), stream=stream0)
        buf1102 = empty((1, 96), device='cuda', dtype=torch.float32)
        buf1103 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_86.run(buf1098, buf1099, alias_274, alias_275, buf1102, buf1103, 96, 8, grid=grid(96), stream=stream0)
        buf1104 = buf1093; del buf1093  # reuse
        buf1105 = buf3; del buf3  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_90.run(buf1104, buf1094, buf1097, alias_275, primals_19, buf2, buf1101, alias_274, buf1100, primals_18, buf1105, 2408448, grid=grid(2408448), stream=stream0)
        del alias_274
        del alias_275
        del buf1094
        del buf1097
        del primals_18
        del primals_19
        buf1106 = reinterpret_tensor(buf1095, (1, 96, 1, 1, 196), (18816, 196, 18816, 18816, 1), 0); del buf1095  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_75.run(buf1104, convolution_6, buf1106, 18816, 128, grid=grid(18816), stream=stream0)
        del convolution_6
        buf1107 = empty((1, 96, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_76.run(buf1106, buf1107, 96, 196, grid=grid(96), stream=stream0)
        buf1108 = empty_strided((96, 4), (1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_77.run(buf1105, buf1108, 384, 6272, grid=grid(384), stream=stream0)
        buf1109 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_78.run(buf1108, buf1109, 96, 4, grid=grid(96), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1110 = aten.convolution_backward(buf1105, clone_4, primals_231, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_4
        del primals_231
        buf1111 = buf1110[0]
        buf1112 = buf1110[1]
        del buf1110
        buf1113 = buf1111; del buf1111  # reuse
        # Source Nodes: [x_22], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_79.run(buf1113, convolution_5, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del convolution_5
        buf1114 = reinterpret_tensor(buf1108, (384, ), (1, ), 0); del buf1108  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_80.run(buf1113, buf1114, 384, 25088, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1115 = aten.convolution_backward(buf1113, add_18, primals_229, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_18
        del buf1113
        del primals_229
        buf1116 = buf1115[0]
        buf1117 = buf1115[1]
        del buf1115
        buf1118 = buf1085; del buf1085  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_81.run(buf1116, buf1, avg_pool2d_2, add_15, primals_15, buf1118, 19200, 126, grid=grid(19200), stream=stream0)
        buf1119 = buf1099; del buf1099  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_82.run(buf1118, buf1119, 768, 25, grid=grid(768), stream=stream0)
        buf1120 = buf1098; del buf1098  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_83.run(buf1116, buf1120, 768, 3136, grid=grid(768), stream=stream0)
        buf1121 = buf1101; del buf1101  # reuse
        buf1122 = buf1100; del buf1100  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_84.run(buf1119, primals_16, buf1120, buf1121, buf1122, 8, 96, grid=grid(8), stream=stream0)
        buf1123 = reinterpret_tensor(buf1116, (8, 1, 96, 3136), (301056, 2408448, 3136, 1), 0); del buf1116  # reuse
        buf1126 = buf1104; del buf1104  # reuse
        buf1127 = buf1105; del buf1105  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_85.run(buf1123, buf1126, alias_277, primals_16, buf1, avg_pool2d_2, add_15, primals_15, buf1122, alias_276, buf1121, buf1127, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_15
        del primals_16
        buf1124 = empty((1, 96), device='cuda', dtype=torch.float32)
        buf1125 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_86.run(buf1119, buf1120, alias_276, alias_277, buf1124, buf1125, 96, 8, grid=grid(96), stream=stream0)
        del alias_276
        del alias_277
        buf1128 = reinterpret_tensor(buf1106, (1, 96, 1, 1, 196), (18816, 1, 18816, 18816, 96), 0); del buf1106  # reuse
        # Source Nodes: [sub_2], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_87.run(buf1126, avg_pool2d_2, add_15, buf1128, 18816, 128, grid=grid(18816), stream=stream0)
        del add_15
        del avg_pool2d_2
        buf1129 = empty((1, 96, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_2], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_88.run(buf1128, buf1129, 96, 196, grid=grid(96), stream=stream0)
        buf1130 = reinterpret_tensor(buf1123, (8, 96, 56, 56), (301056, 3136, 56, 1), 0); del buf1123  # reuse
        buf1131 = buf1120; del buf1120  # reuse
        buf1132 = buf1119; del buf1119  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_red_fused_avg_pool2d_backward_native_group_norm_backward_89.run(buf1127, buf1, buf1130, buf1131, buf1132, 768, 3136, grid=grid(768), stream=stream0)
        buf1133 = buf1122; del buf1122  # reuse
        buf1134 = buf1121; del buf1121  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_84.run(buf1131, primals_13, buf1132, buf1133, buf1134, 8, 96, grid=grid(8), stream=stream0)
        buf1135 = empty((1, 96), device='cuda', dtype=torch.float32)
        buf1136 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_86.run(buf1131, buf1132, alias_278, alias_279, buf1135, buf1136, 96, 8, grid=grid(96), stream=stream0)
        buf1137 = buf1; del buf1  # reuse
        buf1138 = buf2; del buf2  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_91.run(buf1137, buf1126, buf1127, buf1130, alias_279, primals_13, buf1134, alias_278, buf1133, primals_12, buf1138, 2408448, grid=grid(2408448), stream=stream0)
        del alias_278
        del alias_279
        del buf1126
        del buf1127
        del primals_12
        del primals_13
        buf1139 = reinterpret_tensor(buf1128, (1, 96, 1, 1, 196), (18816, 196, 18816, 18816, 1), 0); del buf1128  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_75.run(buf1137, convolution_4, buf1139, 18816, 128, grid=grid(18816), stream=stream0)
        del convolution_4
        buf1140 = empty((1, 96, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_76.run(buf1139, buf1140, 96, 196, grid=grid(96), stream=stream0)
        buf1141 = empty_strided((96, 4), (1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_77.run(buf1138, buf1141, 384, 6272, grid=grid(384), stream=stream0)
        buf1142 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_78.run(buf1141, buf1142, 96, 4, grid=grid(96), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1143 = aten.convolution_backward(buf1138, clone_2, primals_227, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_2
        del primals_227
        buf1144 = buf1143[0]
        buf1145 = buf1143[1]
        del buf1143
        buf1146 = buf1144; del buf1144  # reuse
        # Source Nodes: [x_14], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_79.run(buf1146, convolution_3, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del convolution_3
        buf1147 = reinterpret_tensor(buf1141, (384, ), (1, ), 0); del buf1141  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_80.run(buf1146, buf1147, 384, 25088, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1148 = aten.convolution_backward(buf1146, add_11, primals_225, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_11
        del buf1146
        del primals_225
        buf1149 = buf1148[0]
        buf1150 = buf1148[1]
        del buf1148
        buf1151 = buf1118; del buf1118  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_81.run(buf1149, buf0, avg_pool2d_1, add_8, primals_9, buf1151, 19200, 126, grid=grid(19200), stream=stream0)
        buf1152 = buf1132; del buf1132  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_82.run(buf1151, buf1152, 768, 25, grid=grid(768), stream=stream0)
        buf1153 = buf1131; del buf1131  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_83.run(buf1149, buf1153, 768, 3136, grid=grid(768), stream=stream0)
        buf1154 = buf1134; del buf1134  # reuse
        buf1155 = buf1133; del buf1133  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_84.run(buf1152, primals_10, buf1153, buf1154, buf1155, 8, 96, grid=grid(8), stream=stream0)
        buf1156 = reinterpret_tensor(buf1149, (8, 1, 96, 3136), (301056, 2408448, 3136, 1), 0); del buf1149  # reuse
        buf1159 = buf1137; del buf1137  # reuse
        buf1160 = buf1138; del buf1138  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_group_norm_backward]
        triton_poi_fused_add_mul_native_group_norm_backward_85.run(buf1156, buf1159, alias_281, primals_10, buf0, avg_pool2d_1, add_8, primals_9, buf1155, alias_280, buf1154, buf1160, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_10
        del primals_9
        buf1157 = empty((1, 96), device='cuda', dtype=torch.float32)
        buf1158 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_86.run(buf1152, buf1153, alias_280, alias_281, buf1157, buf1158, 96, 8, grid=grid(96), stream=stream0)
        del alias_280
        del alias_281
        buf1161 = reinterpret_tensor(buf1139, (1, 96, 1, 1, 196), (18816, 1, 18816, 18816, 96), 0); del buf1139  # reuse
        # Source Nodes: [sub_1], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_87.run(buf1159, avg_pool2d_1, add_8, buf1161, 18816, 128, grid=grid(18816), stream=stream0)
        del add_8
        del avg_pool2d_1
        buf1162 = empty((1, 96, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub_1], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_88.run(buf1161, buf1162, 96, 196, grid=grid(96), stream=stream0)
        buf1163 = reinterpret_tensor(buf1156, (8, 96, 56, 56), (301056, 3136, 56, 1), 0); del buf1156  # reuse
        buf1164 = buf1153; del buf1153  # reuse
        buf1165 = buf1152; del buf1152  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_red_fused_avg_pool2d_backward_native_group_norm_backward_89.run(buf1160, buf0, buf1163, buf1164, buf1165, 768, 3136, grid=grid(768), stream=stream0)
        buf1166 = buf1155; del buf1155  # reuse
        buf1167 = buf1154; del buf1154  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_84.run(buf1164, primals_7, buf1165, buf1166, buf1167, 8, 96, grid=grid(8), stream=stream0)
        buf1168 = empty((1, 96), device='cuda', dtype=torch.float32)
        buf1169 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_86.run(buf1164, buf1165, alias_282, alias_283, buf1168, buf1169, 96, 8, grid=grid(96), stream=stream0)
        buf1170 = buf0; del buf0  # reuse
        buf1171 = buf1130; del buf1130  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_91.run(buf1170, buf1159, buf1160, buf1163, alias_283, primals_7, buf1167, alias_282, buf1166, primals_6, buf1171, 2408448, grid=grid(2408448), stream=stream0)
        del alias_282
        del alias_283
        del buf1159
        del buf1160
        del buf1163
        del primals_6
        del primals_7
        buf1172 = reinterpret_tensor(buf1161, (1, 96, 1, 1, 196), (18816, 196, 18816, 18816, 1), 0); del buf1161  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_75.run(buf1170, convolution_2, buf1172, 18816, 128, grid=grid(18816), stream=stream0)
        del convolution_2
        buf1173 = empty((1, 96, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_76.run(buf1172, buf1173, 96, 196, grid=grid(96), stream=stream0)
        buf1174 = empty_strided((96, 4), (1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_77.run(buf1171, buf1174, 384, 6272, grid=grid(384), stream=stream0)
        buf1175 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_78.run(buf1174, buf1175, 96, 4, grid=grid(96), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1176 = aten.convolution_backward(buf1171, clone, primals_223, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone
        del primals_223
        buf1177 = buf1176[0]
        buf1178 = buf1176[1]
        del buf1176
        buf1179 = buf1177; del buf1177  # reuse
        # Source Nodes: [x_6], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_79.run(buf1179, convolution_1, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del convolution_1
        buf1180 = reinterpret_tensor(buf1174, (384, ), (1, ), 0); del buf1174  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_80.run(buf1179, buf1180, 384, 25088, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1181 = aten.convolution_backward(buf1179, add_4, primals_221, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_4
        del buf1179
        del primals_221
        buf1182 = buf1181[0]
        buf1183 = buf1181[1]
        del buf1181
        buf1184 = reinterpret_tensor(buf1151, (8, 96, 25), (2400, 1, 96), 0); del buf1151  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_92.run(buf1182, convolution, avg_pool2d, add_1, primals_3, buf1184, 19200, 126, grid=grid(19200), stream=stream0)
        buf1185 = buf1165; del buf1165  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_93.run(buf1184, buf1185, 768, 25, grid=grid(768), stream=stream0)
        del buf1184
        buf1186 = buf1164; del buf1164  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_red_fused_native_group_norm_backward_83.run(buf1182, buf1186, 768, 3136, grid=grid(768), stream=stream0)
        buf1187 = buf1167; del buf1167  # reuse
        buf1188 = buf1166; del buf1166  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_84.run(buf1185, primals_4, buf1186, buf1187, buf1188, 8, 96, grid=grid(8), stream=stream0)
        buf1189 = reinterpret_tensor(buf1171, (8, 1, 96, 3136), (301056, 2408448, 1, 96), 0); del buf1171  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_poi_fused_native_group_norm_backward_94.run(buf1182, alias_285, primals_4, convolution, avg_pool2d, add_1, primals_3, buf1188, alias_284, buf1187, buf1189, 25088, 96, grid=grid(25088, 96), stream=stream0)
        del primals_4
        buf1190 = empty((1, 96), device='cuda', dtype=torch.float32)
        buf1191 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_86.run(buf1185, buf1186, alias_284, alias_285, buf1190, buf1191, 96, 8, grid=grid(96), stream=stream0)
        buf1192 = buf1170; del buf1170  # reuse
        buf1193 = buf1182; del buf1182  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_95.run(buf1192, buf1189, buf1188, alias_284, buf1187, alias_285, primals_3, buf1193, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del alias_284
        del alias_285
        del primals_3
        buf1194 = reinterpret_tensor(buf1172, (1, 96, 1, 1, 196), (18816, 1, 18816, 18816, 96), 0); del buf1172  # reuse
        # Source Nodes: [sub], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_87.run(buf1192, avg_pool2d, add_1, buf1194, 18816, 128, grid=grid(18816), stream=stream0)
        del add_1
        del avg_pool2d
        buf1195 = empty((1, 96, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sub], Original ATen: [aten.mul, aten.sub, aten.sum]
        triton_red_fused_mul_sub_sum_88.run(buf1194, buf1195, 96, 196, grid=grid(96), stream=stream0)
        del buf1194
        buf1196 = reinterpret_tensor(buf1189, (8, 96, 56, 56), (301056, 3136, 56, 1), 0); del buf1189  # reuse
        buf1197 = buf1186; del buf1186  # reuse
        buf1198 = buf1185; del buf1185  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.native_group_norm_backward]
        triton_red_fused_avg_pool2d_backward_native_group_norm_backward_96.run(buf1193, convolution, buf1196, buf1197, buf1198, 768, 3136, grid=grid(768), stream=stream0)
        buf1199 = buf1188; del buf1188  # reuse
        buf1200 = buf1187; del buf1187  # reuse
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_84.run(buf1197, primals_1, buf1198, buf1199, buf1200, 8, 96, grid=grid(8), stream=stream0)
        buf1201 = empty((1, 96), device='cuda', dtype=torch.float32)
        buf1202 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_group_norm_backward]
        triton_per_fused_native_group_norm_backward_86.run(buf1197, buf1198, alias_286, alias_287, buf1201, buf1202, 96, 8, grid=grid(96), stream=stream0)
        del buf1197
        del buf1198
        buf1203 = buf1192; del buf1192  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_97.run(buf1203, buf1193, buf1196, alias_287, primals_1, convolution, buf1200, alias_286, buf1199, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del alias_286
        del alias_287
        del buf1193
        del buf1196
        del buf1199
        del buf1200
        del convolution
        del primals_1
        buf1204 = empty_strided((96, 4), (1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_77.run(buf1203, buf1204, 384, 6272, grid=grid(384), stream=stream0)
        buf1205 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_78.run(buf1204, buf1205, 96, 4, grid=grid(96), stream=stream0)
        del buf1204
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1206 = aten.convolution_backward(buf1203, primals_373, primals_219, [96], [4, 4], [2, 2], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf1203
        del primals_219
        del primals_373
        buf1207 = buf1206[1]
        return (reinterpret_tensor(buf1201, (96, ), (1, ), 0), buf1202, reinterpret_tensor(buf1195, (96, ), (1, ), 0), reinterpret_tensor(buf1190, (96, ), (1, ), 0), buf1191, reinterpret_tensor(buf1173, (96, ), (1, ), 0), reinterpret_tensor(buf1168, (96, ), (1, ), 0), buf1169, reinterpret_tensor(buf1162, (96, ), (1, ), 0), reinterpret_tensor(buf1157, (96, ), (1, ), 0), buf1158, reinterpret_tensor(buf1140, (96, ), (1, ), 0), reinterpret_tensor(buf1135, (96, ), (1, ), 0), buf1136, reinterpret_tensor(buf1129, (96, ), (1, ), 0), reinterpret_tensor(buf1124, (96, ), (1, ), 0), buf1125, reinterpret_tensor(buf1107, (96, ), (1, ), 0), reinterpret_tensor(buf1102, (96, ), (1, ), 0), buf1103, reinterpret_tensor(buf1096, (96, ), (1, ), 0), reinterpret_tensor(buf1091, (96, ), (1, ), 0), buf1092, reinterpret_tensor(buf1074, (96, ), (1, ), 0), reinterpret_tensor(buf1069, (96, ), (1, ), 0), buf1070, reinterpret_tensor(buf1063, (96, ), (1, ), 0), reinterpret_tensor(buf1058, (96, ), (1, ), 0), buf1059, reinterpret_tensor(buf1041, (96, ), (1, ), 0), reinterpret_tensor(buf1036, (96, ), (1, ), 0), buf1037, reinterpret_tensor(buf1030, (96, ), (1, ), 0), reinterpret_tensor(buf1025, (96, ), (1, ), 0), buf1026, reinterpret_tensor(buf1008, (96, ), (1, ), 0), reinterpret_tensor(buf999, (192, ), (1, ), 0), buf1000, reinterpret_tensor(buf993, (192, ), (1, ), 0), reinterpret_tensor(buf988, (192, ), (1, ), 0), buf989, reinterpret_tensor(buf972, (192, ), (1, ), 0), reinterpret_tensor(buf967, (192, ), (1, ), 0), buf968, reinterpret_tensor(buf961, (192, ), (1, ), 0), reinterpret_tensor(buf956, (192, ), (1, ), 0), buf957, reinterpret_tensor(buf940, (192, ), (1, ), 0), reinterpret_tensor(buf935, (192, ), (1, ), 0), buf936, reinterpret_tensor(buf929, (192, ), (1, ), 0), reinterpret_tensor(buf924, (192, ), (1, ), 0), buf925, reinterpret_tensor(buf908, (192, ), (1, ), 0), reinterpret_tensor(buf903, (192, ), (1, ), 0), buf904, reinterpret_tensor(buf897, (192, ), (1, ), 0), reinterpret_tensor(buf892, (192, ), (1, ), 0), buf893, reinterpret_tensor(buf876, (192, ), (1, ), 0), reinterpret_tensor(buf871, (192, ), (1, ), 0), buf872, reinterpret_tensor(buf865, (192, ), (1, ), 0), reinterpret_tensor(buf860, (192, ), (1, ), 0), buf861, reinterpret_tensor(buf844, (192, ), (1, ), 0), reinterpret_tensor(buf839, (192, ), (1, ), 0), buf840, reinterpret_tensor(buf833, (192, ), (1, ), 0), reinterpret_tensor(buf828, (192, ), (1, ), 0), buf829, reinterpret_tensor(buf812, (192, ), (1, ), 0), reinterpret_tensor(buf803, (384, ), (1, ), 0), buf804, reinterpret_tensor(buf797, (384, ), (1, ), 0), reinterpret_tensor(buf792, (384, ), (1, ), 0), buf793, reinterpret_tensor(buf776, (384, ), (1, ), 0), reinterpret_tensor(buf771, (384, ), (1, ), 0), buf772, reinterpret_tensor(buf765, (384, ), (1, ), 0), reinterpret_tensor(buf760, (384, ), (1, ), 0), buf761, reinterpret_tensor(buf744, (384, ), (1, ), 0), reinterpret_tensor(buf739, (384, ), (1, ), 0), buf740, reinterpret_tensor(buf733, (384, ), (1, ), 0), reinterpret_tensor(buf728, (384, ), (1, ), 0), buf729, reinterpret_tensor(buf712, (384, ), (1, ), 0), reinterpret_tensor(buf707, (384, ), (1, ), 0), buf708, reinterpret_tensor(buf701, (384, ), (1, ), 0), reinterpret_tensor(buf696, (384, ), (1, ), 0), buf697, reinterpret_tensor(buf680, (384, ), (1, ), 0), reinterpret_tensor(buf675, (384, ), (1, ), 0), buf676, reinterpret_tensor(buf669, (384, ), (1, ), 0), reinterpret_tensor(buf664, (384, ), (1, ), 0), buf665, reinterpret_tensor(buf648, (384, ), (1, ), 0), reinterpret_tensor(buf643, (384, ), (1, ), 0), buf644, reinterpret_tensor(buf637, (384, ), (1, ), 0), reinterpret_tensor(buf632, (384, ), (1, ), 0), buf633, reinterpret_tensor(buf616, (384, ), (1, ), 0), reinterpret_tensor(buf611, (384, ), (1, ), 0), buf612, reinterpret_tensor(buf605, (384, ), (1, ), 0), reinterpret_tensor(buf600, (384, ), (1, ), 0), buf601, reinterpret_tensor(buf584, (384, ), (1, ), 0), reinterpret_tensor(buf579, (384, ), (1, ), 0), buf580, reinterpret_tensor(buf573, (384, ), (1, ), 0), reinterpret_tensor(buf568, (384, ), (1, ), 0), buf569, reinterpret_tensor(buf552, (384, ), (1, ), 0), reinterpret_tensor(buf547, (384, ), (1, ), 0), buf548, reinterpret_tensor(buf541, (384, ), (1, ), 0), reinterpret_tensor(buf536, (384, ), (1, ), 0), buf537, reinterpret_tensor(buf520, (384, ), (1, ), 0), reinterpret_tensor(buf515, (384, ), (1, ), 0), buf516, reinterpret_tensor(buf509, (384, ), (1, ), 0), reinterpret_tensor(buf504, (384, ), (1, ), 0), buf505, reinterpret_tensor(buf488, (384, ), (1, ), 0), reinterpret_tensor(buf483, (384, ), (1, ), 0), buf484, reinterpret_tensor(buf477, (384, ), (1, ), 0), reinterpret_tensor(buf472, (384, ), (1, ), 0), buf473, reinterpret_tensor(buf456, (384, ), (1, ), 0), reinterpret_tensor(buf451, (384, ), (1, ), 0), buf452, reinterpret_tensor(buf445, (384, ), (1, ), 0), reinterpret_tensor(buf440, (384, ), (1, ), 0), buf441, reinterpret_tensor(buf424, (384, ), (1, ), 0), reinterpret_tensor(buf419, (384, ), (1, ), 0), buf420, reinterpret_tensor(buf413, (384, ), (1, ), 0), reinterpret_tensor(buf408, (384, ), (1, ), 0), buf409, reinterpret_tensor(buf392, (384, ), (1, ), 0), reinterpret_tensor(buf387, (384, ), (1, ), 0), buf388, reinterpret_tensor(buf381, (384, ), (1, ), 0), reinterpret_tensor(buf376, (384, ), (1, ), 0), buf377, reinterpret_tensor(buf360, (384, ), (1, ), 0), reinterpret_tensor(buf355, (384, ), (1, ), 0), buf356, reinterpret_tensor(buf349, (384, ), (1, ), 0), reinterpret_tensor(buf344, (384, ), (1, ), 0), buf345, reinterpret_tensor(buf328, (384, ), (1, ), 0), reinterpret_tensor(buf323, (384, ), (1, ), 0), buf324, reinterpret_tensor(buf317, (384, ), (1, ), 0), reinterpret_tensor(buf312, (384, ), (1, ), 0), buf313, reinterpret_tensor(buf296, (384, ), (1, ), 0), reinterpret_tensor(buf291, (384, ), (1, ), 0), buf292, reinterpret_tensor(buf285, (384, ), (1, ), 0), reinterpret_tensor(buf280, (384, ), (1, ), 0), buf281, reinterpret_tensor(buf264, (384, ), (1, ), 0), reinterpret_tensor(buf259, (384, ), (1, ), 0), buf260, reinterpret_tensor(buf253, (384, ), (1, ), 0), reinterpret_tensor(buf248, (384, ), (1, ), 0), buf249, reinterpret_tensor(buf232, (384, ), (1, ), 0), reinterpret_tensor(buf223, (768, ), (1, ), 0), buf224, reinterpret_tensor(buf217, (768, ), (1, ), 0), reinterpret_tensor(buf212, (768, ), (1, ), 0), buf213, reinterpret_tensor(buf197, (768, ), (1, ), 0), reinterpret_tensor(buf192, (768, ), (1, ), 0), buf193, reinterpret_tensor(buf186, (768, ), (1, ), 0), reinterpret_tensor(buf181, (768, ), (1, ), 0), buf182, reinterpret_tensor(buf166, (768, ), (1, ), 0), reinterpret_tensor(buf161, (768, ), (1, ), 0), buf162, reinterpret_tensor(buf155, (768, ), (1, ), 0), reinterpret_tensor(buf150, (768, ), (1, ), 0), buf151, reinterpret_tensor(buf135, (768, ), (1, ), 0), reinterpret_tensor(buf130, (768, ), (1, ), 0), buf131, reinterpret_tensor(buf124, (768, ), (1, ), 0), reinterpret_tensor(buf119, (768, ), (1, ), 0), buf120, reinterpret_tensor(buf104, (768, ), (1, ), 0), reinterpret_tensor(buf99, (768, ), (1, ), 0), buf100, reinterpret_tensor(buf93, (768, ), (1, ), 0), reinterpret_tensor(buf88, (768, ), (1, ), 0), buf89, reinterpret_tensor(buf73, (768, ), (1, ), 0), reinterpret_tensor(buf68, (768, ), (1, ), 0), buf69, reinterpret_tensor(buf62, (768, ), (1, ), 0), reinterpret_tensor(buf57, (768, ), (1, ), 0), buf58, reinterpret_tensor(buf42, (768, ), (1, ), 0), buf37, buf38, buf1207, buf1205, buf1183, buf1180, buf1178, buf1175, buf1150, buf1147, buf1145, buf1142, buf1117, buf1114, buf1112, buf1109, buf1084, buf1081, buf1079, buf1076, buf1051, buf1048, buf1046, buf1043, buf1018, buf1015, buf1013, buf1010, buf1005, buf1002, buf981, buf978, buf976, buf973, buf949, buf946, buf944, buf941, buf917, buf914, buf912, buf909, buf885, buf882, buf880, buf877, buf853, buf850, buf848, buf845, buf821, buf818, buf816, buf813, buf809, buf806, buf785, buf782, buf780, buf777, buf753, buf750, buf748, buf745, buf721, buf718, buf716, buf713, buf689, buf686, buf684, buf681, buf657, buf654, buf652, buf649, buf625, buf622, buf620, buf617, buf593, buf590, buf588, buf585, buf561, buf558, buf556, buf553, buf529, buf526, buf524, buf521, buf497, buf494, buf492, buf489, buf465, buf462, buf460, buf457, buf433, buf430, buf428, buf425, buf401, buf398, buf396, buf393, buf369, buf366, buf364, buf361, buf337, buf334, buf332, buf329, buf305, buf302, buf300, buf297, buf273, buf270, buf268, buf265, buf241, buf238, buf236, buf233, buf229, buf226, buf206, buf203, buf201, buf198, buf175, buf172, buf170, buf167, buf144, buf141, buf139, buf136, buf113, buf110, buf108, buf105, buf82, buf79, buf77, buf74, buf51, buf48, buf46, buf43, reinterpret_tensor(buf33, (1000, 768), (768, 1), 0), reinterpret_tensor(buf34, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((96, 3, 7, 7), (147, 1, 21, 3), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((192, 96, 3, 3), (864, 1, 288, 96), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((384, 192, 3, 3), (1728, 1, 576, 192), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((768, 384, 3, 3), (3456, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    add_1 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    avg_pool2d = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    add_4 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    add_8 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    avg_pool2d_1 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    add_11 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda:0', dtype=torch.float32)
    clone_2 = rand_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    add_15 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    avg_pool2d_2 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    add_18 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda:0', dtype=torch.float32)
    clone_4 = rand_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    add_22 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    avg_pool2d_3 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    add_25 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda:0', dtype=torch.float32)
    clone_6 = rand_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    add_29 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    avg_pool2d_4 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    add_32 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda:0', dtype=torch.float32)
    clone_8 = rand_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    add_36 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    avg_pool2d_5 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    add_39 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda:0', dtype=torch.float32)
    clone_10 = rand_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    add_41 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    add_43 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    avg_pool2d_6 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    add_46 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    clone_12 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    add_50 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    avg_pool2d_7 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    add_53 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    clone_14 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    add_57 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    avg_pool2d_8 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    add_60 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    clone_16 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    add_64 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    avg_pool2d_9 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    add_67 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    clone_18 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    add_71 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    avg_pool2d_10 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    add_74 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    clone_20 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    add_78 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    avg_pool2d_11 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    add_81 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    clone_22 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    add_83 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_85 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    avg_pool2d_12 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_88 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    clone_24 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_92 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    avg_pool2d_13 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_95 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    clone_26 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_99 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    avg_pool2d_14 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_102 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    clone_28 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_106 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    avg_pool2d_15 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_109 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    clone_30 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_113 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    avg_pool2d_16 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_116 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    clone_32 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_120 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    avg_pool2d_17 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_123 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    clone_34 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_127 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    avg_pool2d_18 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_130 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    clone_36 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_134 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    avg_pool2d_19 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_137 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    clone_38 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_141 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    avg_pool2d_20 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_144 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    clone_40 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_148 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    avg_pool2d_21 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_151 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    clone_42 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_155 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    avg_pool2d_22 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_158 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    clone_44 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_162 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    avg_pool2d_23 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_165 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    clone_46 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_169 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    avg_pool2d_24 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_172 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    clone_48 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_176 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    avg_pool2d_25 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_179 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    clone_50 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_183 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    avg_pool2d_26 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_186 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    clone_52 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_190 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    avg_pool2d_27 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_193 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    clone_54 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_197 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    avg_pool2d_28 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_200 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_59 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    clone_56 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_204 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    avg_pool2d_29 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_207 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    clone_58 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    add_209 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_63 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    add_211 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    avg_pool2d_30 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    add_214 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    convolution_64 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cuda:0', dtype=torch.float32)
    clone_60 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cuda:0', dtype=torch.float32)
    convolution_65 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    add_218 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    avg_pool2d_31 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    add_221 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    convolution_66 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cuda:0', dtype=torch.float32)
    clone_62 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cuda:0', dtype=torch.float32)
    convolution_67 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    add_225 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    avg_pool2d_32 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    add_228 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    convolution_68 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cuda:0', dtype=torch.float32)
    clone_64 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cuda:0', dtype=torch.float32)
    convolution_69 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    add_232 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    avg_pool2d_33 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    add_235 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    convolution_70 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cuda:0', dtype=torch.float32)
    clone_66 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cuda:0', dtype=torch.float32)
    convolution_71 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    add_239 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    avg_pool2d_34 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    add_242 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    convolution_72 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cuda:0', dtype=torch.float32)
    clone_68 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cuda:0', dtype=torch.float32)
    convolution_73 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    add_246 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    avg_pool2d_35 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    add_249 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    convolution_74 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cuda:0', dtype=torch.float32)
    clone_70 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cuda:0', dtype=torch.float32)
    convolution_75 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    mul_324 = rand_strided((8, 1, 1, 768), (768, 1, 768, 1), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((8, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_3 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    alias_144 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_145 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_146 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_147 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_148 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_149 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_150 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_151 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_152 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_153 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_154 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_155 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_156 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_157 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_158 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_159 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_160 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_161 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_162 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_163 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_164 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_165 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_166 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_167 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_168 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_169 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_170 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_171 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_172 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_173 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_174 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_175 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_176 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_177 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_178 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_179 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_180 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_181 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_182 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_183 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_184 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_185 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_186 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_187 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_188 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_189 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_190 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_191 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_192 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_193 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_194 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_195 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_196 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_197 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_198 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_199 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_200 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_201 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_202 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_203 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_204 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_205 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_206 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_207 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_208 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_209 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_210 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_211 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_212 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_213 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_214 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_215 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_216 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_217 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_218 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_219 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_220 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_221 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_222 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_223 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_224 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_225 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_226 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_227 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_228 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_229 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_230 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_231 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_232 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_233 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_234 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_235 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_236 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_237 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_238 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_239 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_240 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_241 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_242 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_243 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_244 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_245 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_246 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_247 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_248 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_249 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_250 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_251 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_252 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_253 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_254 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_255 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_256 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_257 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_258 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_259 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_260 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_261 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_262 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_263 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_264 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_265 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_266 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_267 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_268 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_269 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_270 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_271 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_272 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_273 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_274 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_275 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_276 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_277 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_278 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_279 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_280 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_281 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_282 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_283 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_284 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_285 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_286 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    alias_287 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_141, primals_142, primals_144, primals_145, primals_147, primals_148, primals_150, primals_151, primals_153, primals_154, primals_156, primals_157, primals_159, primals_160, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_221, primals_223, primals_225, primals_227, primals_229, primals_231, primals_233, primals_235, primals_237, primals_239, primals_241, primals_243, primals_245, primals_247, primals_249, primals_251, primals_253, primals_255, primals_257, primals_259, primals_261, primals_263, primals_265, primals_267, primals_269, primals_271, primals_273, primals_275, primals_277, primals_279, primals_281, primals_283, primals_285, primals_287, primals_289, primals_291, primals_293, primals_295, primals_297, primals_299, primals_301, primals_303, primals_305, primals_307, primals_309, primals_311, primals_313, primals_315, primals_317, primals_319, primals_321, primals_323, primals_325, primals_327, primals_329, primals_331, primals_333, primals_335, primals_337, primals_339, primals_341, primals_343, primals_345, primals_347, primals_349, primals_351, primals_353, primals_355, primals_357, primals_359, primals_361, primals_363, primals_365, primals_367, primals_369, primals_373, convolution, add_1, avg_pool2d, add_4, convolution_1, clone, convolution_2, add_8, avg_pool2d_1, add_11, convolution_3, clone_2, convolution_4, add_15, avg_pool2d_2, add_18, convolution_5, clone_4, convolution_6, add_22, avg_pool2d_3, add_25, convolution_7, clone_6, convolution_8, add_29, avg_pool2d_4, add_32, convolution_9, clone_8, convolution_10, add_36, avg_pool2d_5, add_39, convolution_11, clone_10, convolution_12, add_41, convolution_13, add_43, avg_pool2d_6, add_46, convolution_14, clone_12, convolution_15, add_50, avg_pool2d_7, add_53, convolution_16, clone_14, convolution_17, add_57, avg_pool2d_8, add_60, convolution_18, clone_16, convolution_19, add_64, avg_pool2d_9, add_67, convolution_20, clone_18, convolution_21, add_71, avg_pool2d_10, add_74, convolution_22, clone_20, convolution_23, add_78, avg_pool2d_11, add_81, convolution_24, clone_22, convolution_25, add_83, convolution_26, add_85, avg_pool2d_12, add_88, convolution_27, clone_24, convolution_28, add_92, avg_pool2d_13, add_95, convolution_29, clone_26, convolution_30, add_99, avg_pool2d_14, add_102, convolution_31, clone_28, convolution_32, add_106, avg_pool2d_15, add_109, convolution_33, clone_30, convolution_34, add_113, avg_pool2d_16, add_116, convolution_35, clone_32, convolution_36, add_120, avg_pool2d_17, add_123, convolution_37, clone_34, convolution_38, add_127, avg_pool2d_18, add_130, convolution_39, clone_36, convolution_40, add_134, avg_pool2d_19, add_137, convolution_41, clone_38, convolution_42, add_141, avg_pool2d_20, add_144, convolution_43, clone_40, convolution_44, add_148, avg_pool2d_21, add_151, convolution_45, clone_42, convolution_46, add_155, avg_pool2d_22, add_158, convolution_47, clone_44, convolution_48, add_162, avg_pool2d_23, add_165, convolution_49, clone_46, convolution_50, add_169, avg_pool2d_24, add_172, convolution_51, clone_48, convolution_52, add_176, avg_pool2d_25, add_179, convolution_53, clone_50, convolution_54, add_183, avg_pool2d_26, add_186, convolution_55, clone_52, convolution_56, add_190, avg_pool2d_27, add_193, convolution_57, clone_54, convolution_58, add_197, avg_pool2d_28, add_200, convolution_59, clone_56, convolution_60, add_204, avg_pool2d_29, add_207, convolution_61, clone_58, convolution_62, add_209, convolution_63, add_211, avg_pool2d_30, add_214, convolution_64, clone_60, convolution_65, add_218, avg_pool2d_31, add_221, convolution_66, clone_62, convolution_67, add_225, avg_pool2d_32, add_228, convolution_68, clone_64, convolution_69, add_232, avg_pool2d_33, add_235, convolution_70, clone_66, convolution_71, add_239, avg_pool2d_34, add_242, convolution_72, clone_68, convolution_73, add_246, avg_pool2d_35, add_249, convolution_74, clone_70, convolution_75, mul_324, view_216, permute_3, div, alias_144, alias_145, alias_146, alias_147, alias_148, alias_149, alias_150, alias_151, alias_152, alias_153, alias_154, alias_155, alias_156, alias_157, alias_158, alias_159, alias_160, alias_161, alias_162, alias_163, alias_164, alias_165, alias_166, alias_167, alias_168, alias_169, alias_170, alias_171, alias_172, alias_173, alias_174, alias_175, alias_176, alias_177, alias_178, alias_179, alias_180, alias_181, alias_182, alias_183, alias_184, alias_185, alias_186, alias_187, alias_188, alias_189, alias_190, alias_191, alias_192, alias_193, alias_194, alias_195, alias_196, alias_197, alias_198, alias_199, alias_200, alias_201, alias_202, alias_203, alias_204, alias_205, alias_206, alias_207, alias_208, alias_209, alias_210, alias_211, alias_212, alias_213, alias_214, alias_215, alias_216, alias_217, alias_218, alias_219, alias_220, alias_221, alias_222, alias_223, alias_224, alias_225, alias_226, alias_227, alias_228, alias_229, alias_230, alias_231, alias_232, alias_233, alias_234, alias_235, alias_236, alias_237, alias_238, alias_239, alias_240, alias_241, alias_242, alias_243, alias_244, alias_245, alias_246, alias_247, alias_248, alias_249, alias_250, alias_251, alias_252, alias_253, alias_254, alias_255, alias_256, alias_257, alias_258, alias_259, alias_260, alias_261, alias_262, alias_263, alias_264, alias_265, alias_266, alias_267, alias_268, alias_269, alias_270, alias_271, alias_272, alias_273, alias_274, alias_275, alias_276, alias_277, alias_278, alias_279, alias_280, alias_281, alias_282, alias_283, alias_284, alias_285, alias_286, alias_287, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('poolformer_m36', benchmark_compiled_module)
