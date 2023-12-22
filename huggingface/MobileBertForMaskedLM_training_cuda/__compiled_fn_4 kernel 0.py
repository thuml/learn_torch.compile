
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


# kernel path: /tmp/torchinductor_youkaichao/2c/c2cg5ecivll7d6rxrp3istrbm7irzmcpbzlfpdtnvdezhz5jsbme.py
# Source Nodes: [add_10, add_12, add_6, add_8, attention_output, attention_output_1, attention_output_2, layer_input_4, mul_2, mul_4, mul_5, mul_6], Original ATen: [aten.add, aten.mul]
# add_10 => add_10
# add_12 => add_12
# add_6 => add_6
# add_8 => add_8
# attention_output => add_7
# attention_output_1 => add_9
# attention_output_2 => add_11
# layer_input_4 => add_3
# mul_2 => mul_2
# mul_4 => mul_4
# mul_5 => mul_5
# mul_6 => mul_6
triton_poi_fused_add_mul_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x2), None)
    tmp3 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x2), None)
    tmp14 = tl.load(in_ptr8 + (x2), None)
    tmp15 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp1 + tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 + tmp10
    tmp12 = tmp0 + tmp11
    tmp16 = tmp12 * tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = tmp14 + tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tmp13 + tmp23
    tl.store(out_ptr0 + (x2), tmp12, None)
    tl.store(out_ptr1 + (x2), tmp24, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/l5/cl5l62q6vqteudrfhwukqmay66g5vurfwegy5vbcxlmkfjqcjdht.py
# Source Nodes: [add_106, add_121, add_136, add_151, add_166, add_181, add_196, add_211, add_226, add_241, add_256, add_271, add_286, add_301, add_31, add_316, add_331, add_346, add_46, add_61, add_76, add_91, mul_105, mul_113, mul_121, mul_129, mul_137, mul_145, mul_153, mul_161, mul_169, mul_17, mul_177, mul_25, mul_33, mul_41, mul_49, mul_57, mul_65, mul_73, mul_81, mul_89, mul_9, mul_97, value_tensor_1, value_tensor_10, value_tensor_11, value_tensor_12, value_tensor_13, value_tensor_14, value_tensor_15, value_tensor_16, value_tensor_17, value_tensor_18, value_tensor_19, value_tensor_2, value_tensor_20, value_tensor_21, value_tensor_22, value_tensor_3, value_tensor_4, value_tensor_5, value_tensor_6, value_tensor_7, value_tensor_8, value_tensor_9], Original ATen: [aten.add, aten.mul]
# add_106 => add_106
# add_121 => add_121
# add_136 => add_136
# add_151 => add_151
# add_166 => add_166
# add_181 => add_181
# add_196 => add_196
# add_211 => add_211
# add_226 => add_226
# add_241 => add_241
# add_256 => add_256
# add_271 => add_271
# add_286 => add_286
# add_301 => add_301
# add_31 => add_31
# add_316 => add_316
# add_331 => add_331
# add_346 => add_346
# add_46 => add_46
# add_61 => add_61
# add_76 => add_76
# add_91 => add_91
# mul_105 => mul_105
# mul_113 => mul_113
# mul_121 => mul_121
# mul_129 => mul_129
# mul_137 => mul_137
# mul_145 => mul_145
# mul_153 => mul_153
# mul_161 => mul_161
# mul_169 => mul_169
# mul_17 => mul_17
# mul_177 => mul_177
# mul_25 => mul_25
# mul_33 => mul_33
# mul_41 => mul_41
# mul_49 => mul_49
# mul_57 => mul_57
# mul_65 => mul_65
# mul_73 => mul_73
# mul_81 => mul_81
# mul_89 => mul_89
# mul_9 => mul_9
# mul_97 => mul_97
# value_tensor_1 => add_17
# value_tensor_10 => add_152
# value_tensor_11 => add_167
# value_tensor_12 => add_182
# value_tensor_13 => add_197
# value_tensor_14 => add_212
# value_tensor_15 => add_227
# value_tensor_16 => add_242
# value_tensor_17 => add_257
# value_tensor_18 => add_272
# value_tensor_19 => add_287
# value_tensor_2 => add_32
# value_tensor_20 => add_302
# value_tensor_21 => add_317
# value_tensor_22 => add_332
# value_tensor_3 => add_47
# value_tensor_4 => add_62
# value_tensor_5 => add_77
# value_tensor_6 => add_92
# value_tensor_7 => add_107
# value_tensor_8 => add_122
# value_tensor_9 => add_137
triton_poi_fused_add_mul_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: '*fp32', 78: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(78,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x2), None)
    tmp3 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x2), None)
    tmp14 = tl.load(in_ptr8 + (x2), None)
    tmp15 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr13 + (x2), None)
    tmp26 = tl.load(in_ptr14 + (x2), None)
    tmp27 = tl.load(in_ptr15 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr16 + (x0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr17 + (x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr18 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr19 + (x2), None)
    tmp38 = tl.load(in_ptr20 + (x2), None)
    tmp39 = tl.load(in_ptr21 + (x0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr22 + (x0), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr23 + (x0), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr24 + (x0), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr25 + (x2), None)
    tmp50 = tl.load(in_ptr26 + (x2), None)
    tmp51 = tl.load(in_ptr27 + (x0), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr28 + (x0), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr29 + (x0), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr30 + (x0), None, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr31 + (x2), None)
    tmp62 = tl.load(in_ptr32 + (x2), None)
    tmp63 = tl.load(in_ptr33 + (x0), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr34 + (x0), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr35 + (x0), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr36 + (x0), None, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr37 + (x2), None)
    tmp74 = tl.load(in_ptr38 + (x2), None)
    tmp75 = tl.load(in_ptr39 + (x0), None, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr40 + (x0), None, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr41 + (x0), None, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr42 + (x0), None, eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr43 + (x2), None)
    tmp86 = tl.load(in_ptr44 + (x2), None)
    tmp87 = tl.load(in_ptr45 + (x0), None, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr46 + (x0), None, eviction_policy='evict_last')
    tmp92 = tl.load(in_ptr47 + (x0), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr48 + (x0), None, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr49 + (x2), None)
    tmp98 = tl.load(in_ptr50 + (x2), None)
    tmp99 = tl.load(in_ptr51 + (x0), None, eviction_policy='evict_last')
    tmp101 = tl.load(in_ptr52 + (x0), None, eviction_policy='evict_last')
    tmp104 = tl.load(in_ptr53 + (x0), None, eviction_policy='evict_last')
    tmp106 = tl.load(in_ptr54 + (x0), None, eviction_policy='evict_last')
    tmp109 = tl.load(in_ptr55 + (x2), None)
    tmp110 = tl.load(in_ptr56 + (x2), None)
    tmp111 = tl.load(in_ptr57 + (x0), None, eviction_policy='evict_last')
    tmp113 = tl.load(in_ptr58 + (x0), None, eviction_policy='evict_last')
    tmp116 = tl.load(in_ptr59 + (x0), None, eviction_policy='evict_last')
    tmp118 = tl.load(in_ptr60 + (x0), None, eviction_policy='evict_last')
    tmp121 = tl.load(in_ptr61 + (x2), None)
    tmp122 = tl.load(in_ptr62 + (x2), None)
    tmp123 = tl.load(in_ptr63 + (x0), None, eviction_policy='evict_last')
    tmp125 = tl.load(in_ptr64 + (x0), None, eviction_policy='evict_last')
    tmp128 = tl.load(in_ptr65 + (x0), None, eviction_policy='evict_last')
    tmp130 = tl.load(in_ptr66 + (x0), None, eviction_policy='evict_last')
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp1 + tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 + tmp10
    tmp12 = tmp0 + tmp11
    tmp16 = tmp12 * tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = tmp14 + tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tmp13 + tmp23
    tmp28 = tmp24 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp26 + tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp25 + tmp35
    tmp40 = tmp36 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp38 + tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tmp48 = tmp37 + tmp47
    tmp52 = tmp48 * tmp51
    tmp54 = tmp52 + tmp53
    tmp55 = tmp50 + tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tmp49 + tmp59
    tmp64 = tmp60 * tmp63
    tmp66 = tmp64 + tmp65
    tmp67 = tmp62 + tmp66
    tmp69 = tmp67 * tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = tmp61 + tmp71
    tmp76 = tmp72 * tmp75
    tmp78 = tmp76 + tmp77
    tmp79 = tmp74 + tmp78
    tmp81 = tmp79 * tmp80
    tmp83 = tmp81 + tmp82
    tmp84 = tmp73 + tmp83
    tmp88 = tmp84 * tmp87
    tmp90 = tmp88 + tmp89
    tmp91 = tmp86 + tmp90
    tmp93 = tmp91 * tmp92
    tmp95 = tmp93 + tmp94
    tmp96 = tmp85 + tmp95
    tmp100 = tmp96 * tmp99
    tmp102 = tmp100 + tmp101
    tmp103 = tmp98 + tmp102
    tmp105 = tmp103 * tmp104
    tmp107 = tmp105 + tmp106
    tmp108 = tmp97 + tmp107
    tmp112 = tmp108 * tmp111
    tmp114 = tmp112 + tmp113
    tmp115 = tmp110 + tmp114
    tmp117 = tmp115 * tmp116
    tmp119 = tmp117 + tmp118
    tmp120 = tmp109 + tmp119
    tmp124 = tmp120 * tmp123
    tmp126 = tmp124 + tmp125
    tmp127 = tmp122 + tmp126
    tmp129 = tmp127 * tmp128
    tmp131 = tmp129 + tmp130
    tmp132 = tmp121 + tmp131
    tl.store(out_ptr0 + (x2), tmp12, None)
    tl.store(out_ptr1 + (x2), tmp24, None)
    tl.store(out_ptr2 + (x2), tmp36, None)
    tl.store(out_ptr3 + (x2), tmp48, None)
    tl.store(out_ptr4 + (x2), tmp60, None)
    tl.store(out_ptr5 + (x2), tmp72, None)
    tl.store(out_ptr6 + (x2), tmp84, None)
    tl.store(out_ptr7 + (x2), tmp96, None)
    tl.store(out_ptr8 + (x2), tmp108, None)
    tl.store(out_ptr9 + (x2), tmp120, None)
    tl.store(out_ptr10 + (x2), tmp132, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6s/c6s2qzqxwysi77mkc2uvlyasm3hazhwajjby74nlhua2xeljmuon.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_2
triton_poi_fused_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3906816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r2/cr2gr2e7bgparsn7uvgnc33asfubk56hqv5vaka2tr3l4x7obara.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_2
triton_poi_fused_nll_loss_backward_nll_loss_forward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5y/c5ywvzrdqr3hnhpsx2whvqo3wgsq356mmbkiirkooetauj2xewav.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_3
triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 7631
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4)
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7631*x0)
        tmp1 = tl.full([1, 1], 30522, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r2 + (7631*x0) + (30522*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.full([1, 1], -100, tl.int64)
        tmp6 = tmp4 != tmp5
        tmp11 = tmp8 / tmp10
        tmp12 = 0.0
        tmp13 = tl.where(tmp6, tmp11, tmp12)
        tmp14 = tmp3 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p3/cp3bsb6opzcs23zrbwoo6cyfukdmeriiha262kajunbrcnzbiy2l.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_3
triton_per_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qg/cqgujlj7w6jbzjvl55cin7wno6bvtk3qoo7klqyk6winrywmineh.py
# Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.squeeze]

triton_poi_fused_add_as_strided_scatter_squeeze_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_as_strided_scatter_squeeze_6', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3906816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 30522)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.load(in_ptr4 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp13 = tl.load(in_ptr5 + (x2), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.full([1], -100, tl.int64)
    tmp4 = tmp2 != tmp3
    tmp9 = tmp6 / tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp1 * tmp11
    tmp14 = tl.exp(tmp13)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp12 - tmp16
    tmp18 = tmp0 + tmp17
    tl.store(out_ptr0 + (x2), tmp18, xmask)
    tl.store(out_ptr1 + (x2), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c2/cc2tmgeaofkdiniqpavi3ypnntj6mepcecqabsw5pdeyu27dovzw.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter]

triton_poi_fused_as_strided_scatter_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_7', 'mutated_arg_names': ['out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3906816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/25/c25vdfsi6wtihe5mkg2xjdznb3hxeaonzfsonu5kxusegdt6iydi.py
# Source Nodes: [hidden_states_217, hidden_states_219, loss], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.nll_loss_forward, aten.relu, aten.threshold_backward]
# hidden_states_217 => relu_96
# hidden_states_219 => mul_194, sub_25
# loss => full_default_3
triton_per_fused_native_layer_norm_native_layer_norm_backward_nll_loss_forward_relu_threshold_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_nll_loss_forward_relu_threshold_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = triton_helpers.maximum(0, tmp7)
    tmp10 = tmp8 - tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tmp2 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = 0.0
    tmp19 = tmp8 <= tmp18
    tmp20 = 512.0
    tmp21 = tmp11 / tmp20
    tmp22 = tmp2 * tmp20
    tmp23 = tmp22 - tmp6
    tmp24 = tmp12 * tmp17
    tmp25 = tmp23 - tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tl.where(tmp19, tmp18, tmp26)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uk/cuk6yv45loxtbk7bpsiuxtprh7fa6wddgktuc7ejlnysckrwfa3g.py
# Source Nodes: [], Original ATen: [aten.mul]

triton_poi_fused_mul_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fw/cfwghkm4bvw6g3r3jsuyvkgxx52gx2kocntfibachwdlpdttdmsl.py
# Source Nodes: [], Original ATen: [aten.mul]

triton_poi_fused_mul_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3s/c3sw2htntq4kanej2lrgj7oqnhqwpeptcouonhdmankrgpq4qzlt.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
# loss => full_default_3
triton_poi_fused_nll_loss_forward_threshold_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_forward_threshold_backward_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.int1)
    tmp1 = tl.load(in_out_ptr0 + (x0), None)
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tl.store(in_out_ptr0 + (x0), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ho/cho5bxtcmppl4l6pymrgonvs4uvfzrmcwiy2iot3hco6ort4metq.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/57/c57py4yqplhqshyintecvziuhh2jkngop4l6fmate7yi24ly5xxg.py
# Source Nodes: [add_351, add_353, add_355, add_357, attention_output_115, attention_output_116, attention_output_117, layer_input_119, mul_186, mul_188, mul_189, mul_190], Original ATen: [aten.add, aten.mul]
# add_351 => add_351
# add_353 => add_353
# add_355 => add_355
# add_357 => add_357
# attention_output_115 => add_352
# attention_output_116 => add_354
# attention_output_117 => add_356
# layer_input_119 => add_348
# mul_186 => mul_186
# mul_188 => mul_188
# mul_189 => mul_189
# mul_190 => mul_190
triton_poi_fused_add_mul_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(18,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x2), None)
    tmp3 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x2), None)
    tmp14 = tl.load(in_ptr8 + (x2), None)
    tmp15 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr13 + (x2), None)
    tmp26 = tl.load(in_ptr14 + (x2), None)
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp1 + tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 + tmp10
    tmp12 = tmp0 + tmp11
    tmp16 = tmp12 * tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = tmp14 + tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tmp13 + tmp23
    tmp27 = tmp25 + tmp26
    tmp28 = tmp27 * tmp20
    tl.store(out_ptr0 + (x2), tmp12, None)
    tl.store(out_ptr1 + (x2), tmp24, None)
    tl.store(out_ptr2 + (x2), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lb/clbe5lkqu3ckkxqdfsimpu3k4ehkqqa7asstv24y42yaqqmyvhrj.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 30522
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (30522*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/po/cpolwqailezoxktc6h7wom2e7fwbjfcvjgnntu4c7aegulvecbdc.py
# Source Nodes: [hidden_states_217, hidden_states_219], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.relu]
# hidden_states_217 => relu_96
# hidden_states_219 => mul_194, sub_25
triton_per_fused_native_layer_norm_native_layer_norm_backward_relu_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_relu_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = triton_helpers.maximum(0, tmp1)
    tmp4 = tmp2 - tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iv/civ6fh47jc2n6ynlxe24xssfzwqr6ablivxkaud4y4e3ke6tlqt4.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rr/crrcgn5f232rs7b6pifd2bxofq3eo4ajibamlfyopojrdlq6jozr.py
# Source Nodes: [add_361, mul_185, value_tensor_23], Original ATen: [aten.add, aten.mul, aten.sum]
# add_361 => add_361
# mul_185 => mul_185
# value_tensor_23 => add_347
triton_per_fused_add_mul_sum_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sum_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp5 + tmp10
    tmp12 = tmp0 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kn/cknrk7zgui77wr4fyvkao7jnune3573rsx2uabzidjoc6uas4r6y.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qv/cqvlqmt26e5vtiyp2wuowznujmluevggt4hlze6ayhwzuo3nlbnm.py
# Source Nodes: [add_359, attention_output_118, mul_191], Original ATen: [aten.add, aten.mul, aten.sum]
# add_359 => add_359
# attention_output_118 => add_358
# mul_191 => mul_191
triton_red_fused_add_mul_sum_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr5 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr6 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp7 = tmp5 * tmp6
        tmp9 = tmp7 + tmp8
        tmp10 = tmp4 + tmp9
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp20 = tmp15 + tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
        tmp24 = tmp20 * tmp5
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp17, xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr4 + (x0), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jp/cjptizzopfgdamjimktki2bny4ko5bjs4b452v3cioccnkhchqev.py
# Source Nodes: [add_355, attention_output_116, mul_189], Original ATen: [aten.add, aten.mul, aten.sum]
# add_355 => add_355
# attention_output_116 => add_354
# mul_189 => mul_189
triton_red_fused_add_mul_sum_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr6 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr7 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp5 = tmp0 + tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = tmp9 + tmp14
        tmp16 = tmp5 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
        tmp25 = tmp20 + tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
        tmp29 = tmp25 * tmp10
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask & xmask, tmp32, _tmp31)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp18, xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr4 + (x0), tmp27, xmask)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tl.store(out_ptr5 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tm/ctmeftu55cbyedfwqh6ygq774y55coijue7u4go2qjuox4sdpvxq.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h2/ch27qkvym7cdbkx3swrjmi5hn3gfj73tipdg6lyw2kqu4saugcd3.py
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
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x2), tmp4, None)
    tl.store(out_ptr1 + (x2), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hf/chffkgbugut6ggsoyx2p2qszajcbf5xqlr2k4elnl7jta3tlruyj.py
# Source Nodes: [add_351, layer_input_119, mul_186], Original ATen: [aten.add, aten.mul, aten.sum]
# add_351 => add_351
# layer_input_119 => add_348
# mul_186 => mul_186
triton_red_fused_add_mul_sum_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr6 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp5 = tmp0 + tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = tmp9 + tmp14
        tmp16 = tmp5 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
        tmp24 = tmp20 * tmp10
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp18, xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr4 + (x0), tmp22, xmask)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr5 + (x0), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5e/c5etvh6bhuwxe2wcezclveotuwwoqollbide2bratlk3op2vbqa6.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yj/cyjuam5khwgilmmkkszno32pg65ulyvomk5kx3vz4zjleto7miic.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]

triton_per_fused_add_mul_sum_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sum_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33yrpqtv27xfqbjim5ilpgi2sc6qnngxxngzer7n2ejfy4jxpwk.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]

triton_per_fused_add_mul_sum_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sum_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr4 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp12 = tmp6 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2x/c2xithinf4po6m2va3r3yipqxibgzluxpohfpjeveowun2bwvagb.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x2), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qr/cqr3sm4abtwivutq7wbb4ed66q55cdarm2spmekdvin4zchwkwkz.py
# Source Nodes: [add_331, mul_169, value_tensor_21], Original ATen: [aten.add, aten.mul, aten.sum]
# add_331 => add_331
# mul_169 => mul_169
# value_tensor_21 => add_317
triton_per_fused_add_mul_sum_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sum_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr4 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr5 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp11 + tmp16
    tmp18 = tmp6 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6k/c6kbqqpnu5avulxmipaxfwm3l2cknfuwrjfa4eqkcs2ll4rrtfys.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_29', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x2), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ct/cct33s6xgu4jazivlfekqco3pq3p6i4xapodmbrfklb66h4ups6k.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4v/c4vvjb2p77xofuxsnhztamw7ty2fyxp3vpbg5fkpc5olojgkuxkh.py
# Source Nodes: [loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.mul, aten.nll_loss_forward]
# loss => full_default_3
triton_poi_fused_add_embedding_dense_backward_mul_nll_loss_forward_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_dense_backward_mul_nll_loss_forward_31', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x2), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full([1], False, tl.int1)
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp10, tmp8)
    tmp13 = tl.full([1], -1, tl.int64)
    tmp14 = tmp12 == tmp13
    tmp15 = tl.where(tmp14, tmp10, tmp8)
    tl.store(in_out_ptr0 + (x2), tmp8, None)
    tl.store(out_ptr0 + (x2), tmp11, None)
    tl.store(out_ptr1 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2u/c2ut44gov2yu5zc5g5fz5nnb2we7imfsgygch756wdpdobvlnxix.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zm/czm2oh2hqtom5tnfchxtydyroxmtn2ud6w42jwhimhjm7hp664mq.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ay/caycztwqeliqxuunhqtaac5oyyapsjw5p44scxi4jm336hltb2e2.py
# Source Nodes: [loss], Original ATen: [aten.add, aten.constant_pad_nd, aten.embedding_dense_backward, aten.nll_loss_forward, aten.slice_backward]
# loss => full_default_3
triton_poi_fused_add_constant_pad_nd_embedding_dense_backward_nll_loss_forward_slice_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_embedding_dense_backward_nll_loss_forward_slice_backward_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128)
    x0 = xindex % 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (128 + x0 + (384*x1)), None)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp4 = x1
    tmp5 = tl.full([1], 127, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = 1 + x1
    tmp8 = tmp7 >= tmp1
    tmp9 = tmp8 & tmp6
    tmp10 = tl.load(in_ptr1 + (640 + x0 + (384*x1)), tmp9, other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp6, tmp12, tmp13)
    tmp15 = 0.0
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tmp17 = tmp3 + tmp16
    tmp18 = tl.full([1], 1, tl.int64)
    tmp19 = tmp4 >= tmp18
    tmp20 = (-1) + x1
    tmp21 = tl.full([1], 128, tl.int64)
    tmp22 = tmp20 < tmp21
    tmp23 = tmp22 & tmp19
    tmp24 = tl.load(in_ptr1 + ((-384) + x0 + (384*x1)), tmp23, other=0.0)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp23, tmp24, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp19, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp28, tmp15)
    tmp30 = tmp17 + tmp29
    tmp31 = tl.where(tmp2, tmp15, tmp30)
    tl.store(out_ptr0 + (x2), tmp31, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_21, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_36, primals_37, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_49, primals_50, primals_51, primals_52, primals_53, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_65, primals_66, primals_67, primals_68, primals_69, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_81, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_97, primals_98, primals_99, primals_100, primals_101, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_113, primals_114, primals_115, primals_116, primals_117, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_129, primals_130, primals_131, primals_132, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_145, primals_146, primals_147, primals_148, primals_149, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_161, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_177, primals_178, primals_179, primals_180, primals_181, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_193, primals_194, primals_195, primals_196, primals_197, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_209, primals_210, primals_211, primals_212, primals_213, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_225, primals_226, primals_227, primals_228, primals_229, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_241, primals_242, primals_243, primals_244, primals_245, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_257, primals_258, primals_259, primals_260, primals_261, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_273, primals_274, primals_275, primals_276, primals_277, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_289, primals_290, primals_291, primals_292, primals_293, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_305, primals_306, primals_307, primals_308, primals_309, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_321, primals_322, primals_323, primals_324, primals_325, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_337, primals_338, primals_339, primals_340, primals_341, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_353, primals_354, primals_355, primals_356, primals_357, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_369, primals_370, primals_371, primals_372, primals_373, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_385, primals_1117, primals_1120, primals_1121, full_default, slice_4, view, add_1, view_2, addmm_1, addmm_2, view_6, clone_default_69, clone_default_70, clone_default_71, getitem_212, getitem_213, getitem_214, alias_default_47, view_22, addmm_6, view_24, view_26, addmm_8, view_28, view_30, addmm_10, view_32, view_34, addmm_12, view_36, view_38, addmm_14, view_40, add_16, view_42, addmm_16, addmm_17, view_46, clone_default_66, clone_default_67, clone_default_68, getitem_205, getitem_206, getitem_207, alias_default_45, view_62, addmm_21, view_64, view_66, addmm_23, view_68, view_70, addmm_25, view_72, view_74, addmm_27, view_76, view_78, addmm_29, view_80, addmm_30, view_82, addmm_31, addmm_32, view_86, clone_default_63, clone_default_64, clone_default_65, getitem_198, getitem_199, getitem_200, alias_default_43, view_102, addmm_36, view_104, view_106, addmm_38, view_108, view_110, addmm_40, view_112, view_114, addmm_42, view_116, view_118, addmm_44, view_120, addmm_45, view_122, addmm_46, addmm_47, view_126, clone_default_60, clone_default_61, clone_default_62, getitem_191, getitem_192, getitem_193, alias_default_41, view_142, addmm_51, view_144, view_146, addmm_53, view_148, view_150, addmm_55, view_152, view_154, addmm_57, view_156, view_158, addmm_59, view_160, addmm_60, view_162, addmm_61, addmm_62, view_166, clone_default_57, clone_default_58, clone_default_59, getitem_184, getitem_185, getitem_186, alias_default_39, view_182, addmm_66, view_184, view_186, addmm_68, view_188, view_190, addmm_70, view_192, view_194, addmm_72, view_196, view_198, addmm_74, view_200, addmm_75, view_202, addmm_76, addmm_77, view_206, clone_default_54, clone_default_55, clone_default_56, getitem_177, getitem_178, getitem_179, alias_default_37, view_222, addmm_81, view_224, view_226, addmm_83, view_228, view_230, addmm_85, view_232, view_234, addmm_87, view_236, view_238, addmm_89, view_240, addmm_90, view_242, addmm_91, addmm_92, view_246, clone_default_51, clone_default_52, clone_default_53, getitem_170, getitem_171, getitem_172, alias_default_35, view_262, addmm_96, view_264, view_266, addmm_98, view_268, view_270, addmm_100, view_272, view_274, addmm_102, view_276, view_278, addmm_104, view_280, addmm_105, view_282, addmm_106, addmm_107, view_286, clone_default_48, clone_default_49, clone_default_50, getitem_163, getitem_164, getitem_165, alias_default_33, view_302, addmm_111, view_304, view_306, addmm_113, view_308, view_310, addmm_115, view_312, view_314, addmm_117, view_316, view_318, addmm_119, view_320, addmm_120, view_322, addmm_121, addmm_122, view_326, clone_default_45, clone_default_46, clone_default_47, getitem_156, getitem_157, getitem_158, alias_default_31, view_342, addmm_126, view_344, view_346, addmm_128, view_348, view_350, addmm_130, view_352, view_354, addmm_132, view_356, view_358, addmm_134, view_360, addmm_135, view_362, addmm_136, addmm_137, view_366, clone_default_42, clone_default_43, clone_default_44, getitem_149, getitem_150, getitem_151, alias_default_29, view_382, addmm_141, view_384, view_386, addmm_143, view_388, view_390, addmm_145, view_392, view_394, addmm_147, view_396, view_398, addmm_149, view_400, addmm_150, view_402, addmm_151, addmm_152, view_406, clone_default_39, clone_default_40, clone_default_41, getitem_142, getitem_143, getitem_144, alias_default_27, view_422, addmm_156, view_424, view_426, addmm_158, view_428, view_430, addmm_160, view_432, view_434, addmm_162, view_436, view_438, addmm_164, view_440, addmm_165, view_442, addmm_166, addmm_167, view_446, clone_default_36, clone_default_37, clone_default_38, getitem_135, getitem_136, getitem_137, alias_default_25, view_462, addmm_171, view_464, view_466, addmm_173, view_468, view_470, addmm_175, view_472, view_474, addmm_177, view_476, view_478, addmm_179, view_480, addmm_180, view_482, addmm_181, addmm_182, view_486, clone_default_33, clone_default_34, clone_default_35, getitem_128, getitem_129, getitem_130, alias_default_23, view_502, addmm_186, view_504, view_506, addmm_188, view_508, view_510, addmm_190, view_512, view_514, addmm_192, view_516, view_518, addmm_194, view_520, addmm_195, view_522, addmm_196, addmm_197, view_526, clone_default_30, clone_default_31, clone_default_32, getitem_121, getitem_122, getitem_123, alias_default_21, view_542, addmm_201, view_544, view_546, addmm_203, view_548, view_550, addmm_205, view_552, view_554, addmm_207, view_556, view_558, addmm_209, view_560, addmm_210, view_562, addmm_211, addmm_212, view_566, clone_default_27, clone_default_28, clone_default_29, getitem_114, getitem_115, getitem_116, alias_default_19, view_582, addmm_216, view_584, view_586, addmm_218, view_588, view_590, addmm_220, view_592, view_594, addmm_222, view_596, view_598, addmm_224, view_600, addmm_225, view_602, addmm_226, addmm_227, view_606, clone_default_24, clone_default_25, clone_default_26, getitem_107, getitem_108, getitem_109, alias_default_17, view_622, addmm_231, view_624, view_626, addmm_233, view_628, view_630, addmm_235, view_632, view_634, addmm_237, view_636, view_638, addmm_239, view_640, addmm_240, view_642, addmm_241, addmm_242, view_646, clone_default_21, clone_default_22, clone_default_23, getitem_100, getitem_101, getitem_102, alias_default_15, view_662, addmm_246, view_664, view_666, addmm_248, view_668, view_670, addmm_250, view_672, view_674, addmm_252, view_676, view_678, addmm_254, view_680, addmm_255, view_682, addmm_256, addmm_257, view_686, clone_default_18, clone_default_19, clone_default_20, getitem_93, getitem_94, getitem_95, alias_default_13, view_702, addmm_261, view_704, view_706, addmm_263, view_708, view_710, addmm_265, view_712, view_714, addmm_267, view_716, view_718, addmm_269, view_720, addmm_270, view_722, addmm_271, addmm_272, view_726, clone_default_15, clone_default_16, clone_default_17, getitem_86, getitem_87, getitem_88, alias_default_11, view_742, addmm_276, view_744, view_746, addmm_278, view_748, view_750, addmm_280, view_752, view_754, addmm_282, view_756, view_758, addmm_284, view_760, addmm_285, view_762, addmm_286, addmm_287, view_766, clone_default_12, clone_default_13, clone_default_14, getitem_79, getitem_80, getitem_81, alias_default_9, view_782, addmm_291, view_784, view_786, addmm_293, view_788, view_790, addmm_295, view_792, view_794, addmm_297, view_796, view_798, addmm_299, view_800, addmm_300, view_802, addmm_301, addmm_302, view_806, clone_default_9, clone_default_10, clone_default_11, getitem_72, getitem_73, getitem_74, alias_default_7, view_822, addmm_306, view_824, view_826, addmm_308, view_828, view_830, addmm_310, view_832, view_834, addmm_312, view_836, view_838, addmm_314, view_840, addmm_315, view_842, addmm_316, addmm_317, view_846, clone_default_6, clone_default_7, clone_default_8, getitem_65, getitem_66, getitem_67, alias_default_5, view_862, addmm_321, view_864, view_866, addmm_323, view_868, view_870, addmm_325, view_872, view_874, addmm_327, view_876, view_878, addmm_329, view_880, addmm_330, view_882, addmm_331, addmm_332, view_886, clone_default_3, clone_default_4, clone_default_5, getitem_58, getitem_59, getitem_60, alias_default_3, view_902, addmm_336, view_904, view_906, addmm_338, view_908, view_910, addmm_340, view_912, view_914, addmm_342, view_916, view_918, addmm_344, view_920, addmm_345, view_922, addmm_346, addmm_347, view_926, clone_default, clone_default_1, clone_default_2, getitem_51, getitem_52, getitem_53, alias_default_1, view_942, addmm_351, view_944, view_946, addmm_353, view_948, view_950, addmm_355, view_952, view_954, addmm_357, view_956, view_958, addmm_359, view_960, addmm_360, view_962, addmm_361, getitem_49, rsqrt, sub_27, convert_element_type, permute_483, permute_484, permute_486, permute_490, permute_494, le_1, permute_498, permute_502, le_2, permute_506, permute_510, le_3, permute_514, permute_518, le_4, permute_522, permute_526, permute_539, permute_543, permute_547, permute_551, permute_555, permute_559, permute_563, le_5, permute_567, permute_571, le_6, permute_575, permute_579, le_7, permute_583, permute_587, le_8, permute_591, permute_595, permute_608, permute_612, permute_616, permute_620, permute_624, permute_628, permute_632, le_9, permute_636, permute_640, le_10, permute_644, permute_648, le_11, permute_652, permute_656, le_12, permute_660, permute_664, permute_677, permute_681, permute_685, permute_689, permute_693, permute_697, permute_701, le_13, permute_705, permute_709, le_14, permute_713, permute_717, le_15, permute_721, permute_725, le_16, permute_729, permute_733, permute_746, permute_750, permute_754, permute_758, permute_762, permute_766, permute_770, le_17, permute_774, permute_778, le_18, permute_782, permute_786, le_19, permute_790, permute_794, le_20, permute_798, permute_802, permute_815, permute_819, permute_823, permute_827, permute_831, permute_835, permute_839, le_21, permute_843, permute_847, le_22, permute_851, permute_855, le_23, permute_859, permute_863, le_24, permute_867, permute_871, permute_884, permute_888, permute_892, permute_896, permute_900, permute_904, permute_908, le_25, permute_912, permute_916, le_26, permute_920, permute_924, le_27, permute_928, permute_932, le_28, permute_936, permute_940, permute_953, permute_957, permute_961, permute_965, permute_969, permute_973, permute_977, le_29, permute_981, permute_985, le_30, permute_989, permute_993, le_31, permute_997, permute_1001, le_32, permute_1005, permute_1009, permute_1022, permute_1026, permute_1030, permute_1034, permute_1038, permute_1042, permute_1046, le_33, permute_1050, permute_1054, le_34, permute_1058, permute_1062, le_35, permute_1066, permute_1070, le_36, permute_1074, permute_1078, permute_1091, permute_1095, permute_1099, permute_1103, permute_1107, permute_1111, permute_1115, le_37, permute_1119, permute_1123, le_38, permute_1127, permute_1131, le_39, permute_1135, permute_1139, le_40, permute_1143, permute_1147, permute_1160, permute_1164, permute_1168, permute_1172, permute_1176, permute_1180, permute_1184, le_41, permute_1188, permute_1192, le_42, permute_1196, permute_1200, le_43, permute_1204, permute_1208, le_44, permute_1212, permute_1216, permute_1229, permute_1233, permute_1237, permute_1241, permute_1245, permute_1249, permute_1253, le_45, permute_1257, permute_1261, le_46, permute_1265, permute_1269, le_47, permute_1273, permute_1277, le_48, permute_1281, permute_1285, permute_1298, permute_1302, permute_1306, permute_1310, permute_1314, permute_1318, permute_1322, le_49, permute_1326, permute_1330, le_50, permute_1334, permute_1338, le_51, permute_1342, permute_1346, le_52, permute_1350, permute_1354, permute_1367, permute_1371, permute_1375, permute_1379, permute_1383, permute_1387, permute_1391, le_53, permute_1395, permute_1399, le_54, permute_1403, permute_1407, le_55, permute_1411, permute_1415, le_56, permute_1419, permute_1423, permute_1436, permute_1440, permute_1444, permute_1448, permute_1452, permute_1456, permute_1460, le_57, permute_1464, permute_1468, le_58, permute_1472, permute_1476, le_59, permute_1480, permute_1484, le_60, permute_1488, permute_1492, permute_1505, permute_1509, permute_1513, permute_1517, permute_1521, permute_1525, permute_1529, le_61, permute_1533, permute_1537, le_62, permute_1541, permute_1545, le_63, permute_1549, permute_1553, le_64, permute_1557, permute_1561, permute_1574, permute_1578, permute_1582, permute_1586, permute_1590, permute_1594, permute_1598, le_65, permute_1602, permute_1606, le_66, permute_1610, permute_1614, le_67, permute_1618, permute_1622, le_68, permute_1626, permute_1630, permute_1643, permute_1647, permute_1651, permute_1655, permute_1659, permute_1663, permute_1667, le_69, permute_1671, permute_1675, le_70, permute_1679, permute_1683, le_71, permute_1687, permute_1691, le_72, permute_1695, permute_1699, permute_1712, permute_1716, permute_1720, permute_1724, permute_1728, permute_1732, permute_1736, le_73, permute_1740, permute_1744, le_74, permute_1748, permute_1752, le_75, permute_1756, permute_1760, le_76, permute_1764, permute_1768, permute_1781, permute_1785, permute_1789, permute_1793, permute_1797, permute_1801, permute_1805, le_77, permute_1809, permute_1813, le_78, permute_1817, permute_1821, le_79, permute_1825, permute_1829, le_80, permute_1833, permute_1837, permute_1850, permute_1854, permute_1858, permute_1862, permute_1866, permute_1870, permute_1874, le_81, permute_1878, permute_1882, le_82, permute_1886, permute_1890, le_83, permute_1894, permute_1898, le_84, permute_1902, permute_1906, permute_1919, permute_1923, permute_1927, permute_1931, permute_1935, permute_1939, permute_1943, le_85, permute_1947, permute_1951, le_86, permute_1955, permute_1959, le_87, permute_1963, permute_1967, le_88, permute_1971, permute_1975, permute_1988, permute_1992, permute_1996, permute_2000, permute_2004, permute_2008, permute_2012, le_89, permute_2016, permute_2020, le_90, permute_2024, permute_2028, le_91, permute_2032, permute_2036, le_92, permute_2040, permute_2044, permute_2057, permute_2061, permute_2065, permute_2069, permute_2073, permute_2077, permute_2081, le_93, permute_2085, permute_2089, le_94, permute_2093, permute_2097, le_95, permute_2101, permute_2105, le_96, permute_2109, permute_2113, permute_2126, permute_2130, permute_2134, permute_2138, permute_2142, permute_2146, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_1, (512, ), (1, ))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_17, (512, ), (1, ))
    assert_size_stride(primals_18, (512, ), (1, ))
    assert_size_stride(primals_19, (128, ), (1, ))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (128, ), (1, ))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (128, ), (1, ))
    assert_size_stride(primals_28, (128, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (128, ), (1, ))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (128, ), (1, ))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (128, ), (1, ))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_42, (128, ), (1, ))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_49, (512, ), (1, ))
    assert_size_stride(primals_50, (512, ), (1, ))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (128, ), (1, ))
    assert_size_stride(primals_53, (128, ), (1, ))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_56, (128, ), (1, ))
    assert_size_stride(primals_57, (128, ), (1, ))
    assert_size_stride(primals_58, (128, ), (1, ))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_60, (128, ), (1, ))
    assert_size_stride(primals_61, (128, ), (1, ))
    assert_size_stride(primals_62, (128, ), (1, ))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_66, (512, ), (1, ))
    assert_size_stride(primals_67, (128, ), (1, ))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_69, (128, ), (1, ))
    assert_size_stride(primals_71, (128, ), (1, ))
    assert_size_stride(primals_72, (128, ), (1, ))
    assert_size_stride(primals_73, (128, ), (1, ))
    assert_size_stride(primals_74, (128, ), (1, ))
    assert_size_stride(primals_75, (128, ), (1, ))
    assert_size_stride(primals_76, (128, ), (1, ))
    assert_size_stride(primals_77, (128, ), (1, ))
    assert_size_stride(primals_78, (128, ), (1, ))
    assert_size_stride(primals_79, (128, ), (1, ))
    assert_size_stride(primals_81, (512, ), (1, ))
    assert_size_stride(primals_82, (512, ), (1, ))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (128, ), (1, ))
    assert_size_stride(primals_87, (128, ), (1, ))
    assert_size_stride(primals_88, (128, ), (1, ))
    assert_size_stride(primals_89, (128, ), (1, ))
    assert_size_stride(primals_90, (128, ), (1, ))
    assert_size_stride(primals_91, (128, ), (1, ))
    assert_size_stride(primals_92, (128, ), (1, ))
    assert_size_stride(primals_93, (128, ), (1, ))
    assert_size_stride(primals_94, (128, ), (1, ))
    assert_size_stride(primals_95, (128, ), (1, ))
    assert_size_stride(primals_97, (512, ), (1, ))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_99, (128, ), (1, ))
    assert_size_stride(primals_100, (128, ), (1, ))
    assert_size_stride(primals_101, (128, ), (1, ))
    assert_size_stride(primals_103, (128, ), (1, ))
    assert_size_stride(primals_104, (128, ), (1, ))
    assert_size_stride(primals_105, (128, ), (1, ))
    assert_size_stride(primals_106, (128, ), (1, ))
    assert_size_stride(primals_107, (128, ), (1, ))
    assert_size_stride(primals_108, (128, ), (1, ))
    assert_size_stride(primals_109, (128, ), (1, ))
    assert_size_stride(primals_110, (128, ), (1, ))
    assert_size_stride(primals_111, (128, ), (1, ))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_114, (512, ), (1, ))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_116, (128, ), (1, ))
    assert_size_stride(primals_117, (128, ), (1, ))
    assert_size_stride(primals_119, (128, ), (1, ))
    assert_size_stride(primals_120, (128, ), (1, ))
    assert_size_stride(primals_121, (128, ), (1, ))
    assert_size_stride(primals_122, (128, ), (1, ))
    assert_size_stride(primals_123, (128, ), (1, ))
    assert_size_stride(primals_124, (128, ), (1, ))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (128, ), (1, ))
    assert_size_stride(primals_129, (512, ), (1, ))
    assert_size_stride(primals_130, (512, ), (1, ))
    assert_size_stride(primals_131, (128, ), (1, ))
    assert_size_stride(primals_132, (128, ), (1, ))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_135, (128, ), (1, ))
    assert_size_stride(primals_136, (128, ), (1, ))
    assert_size_stride(primals_137, (128, ), (1, ))
    assert_size_stride(primals_138, (128, ), (1, ))
    assert_size_stride(primals_139, (128, ), (1, ))
    assert_size_stride(primals_140, (128, ), (1, ))
    assert_size_stride(primals_141, (128, ), (1, ))
    assert_size_stride(primals_142, (128, ), (1, ))
    assert_size_stride(primals_143, (128, ), (1, ))
    assert_size_stride(primals_145, (512, ), (1, ))
    assert_size_stride(primals_146, (512, ), (1, ))
    assert_size_stride(primals_147, (128, ), (1, ))
    assert_size_stride(primals_148, (128, ), (1, ))
    assert_size_stride(primals_149, (128, ), (1, ))
    assert_size_stride(primals_151, (128, ), (1, ))
    assert_size_stride(primals_152, (128, ), (1, ))
    assert_size_stride(primals_153, (128, ), (1, ))
    assert_size_stride(primals_154, (128, ), (1, ))
    assert_size_stride(primals_155, (128, ), (1, ))
    assert_size_stride(primals_156, (128, ), (1, ))
    assert_size_stride(primals_157, (128, ), (1, ))
    assert_size_stride(primals_158, (128, ), (1, ))
    assert_size_stride(primals_159, (128, ), (1, ))
    assert_size_stride(primals_161, (512, ), (1, ))
    assert_size_stride(primals_162, (512, ), (1, ))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_164, (128, ), (1, ))
    assert_size_stride(primals_165, (128, ), (1, ))
    assert_size_stride(primals_167, (128, ), (1, ))
    assert_size_stride(primals_168, (128, ), (1, ))
    assert_size_stride(primals_169, (128, ), (1, ))
    assert_size_stride(primals_170, (128, ), (1, ))
    assert_size_stride(primals_171, (128, ), (1, ))
    assert_size_stride(primals_172, (128, ), (1, ))
    assert_size_stride(primals_173, (128, ), (1, ))
    assert_size_stride(primals_174, (128, ), (1, ))
    assert_size_stride(primals_175, (128, ), (1, ))
    assert_size_stride(primals_177, (512, ), (1, ))
    assert_size_stride(primals_178, (512, ), (1, ))
    assert_size_stride(primals_179, (128, ), (1, ))
    assert_size_stride(primals_180, (128, ), (1, ))
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_183, (128, ), (1, ))
    assert_size_stride(primals_184, (128, ), (1, ))
    assert_size_stride(primals_185, (128, ), (1, ))
    assert_size_stride(primals_186, (128, ), (1, ))
    assert_size_stride(primals_187, (128, ), (1, ))
    assert_size_stride(primals_188, (128, ), (1, ))
    assert_size_stride(primals_189, (128, ), (1, ))
    assert_size_stride(primals_190, (128, ), (1, ))
    assert_size_stride(primals_191, (128, ), (1, ))
    assert_size_stride(primals_193, (512, ), (1, ))
    assert_size_stride(primals_194, (512, ), (1, ))
    assert_size_stride(primals_195, (128, ), (1, ))
    assert_size_stride(primals_196, (128, ), (1, ))
    assert_size_stride(primals_197, (128, ), (1, ))
    assert_size_stride(primals_199, (128, ), (1, ))
    assert_size_stride(primals_200, (128, ), (1, ))
    assert_size_stride(primals_201, (128, ), (1, ))
    assert_size_stride(primals_202, (128, ), (1, ))
    assert_size_stride(primals_203, (128, ), (1, ))
    assert_size_stride(primals_204, (128, ), (1, ))
    assert_size_stride(primals_205, (128, ), (1, ))
    assert_size_stride(primals_206, (128, ), (1, ))
    assert_size_stride(primals_207, (128, ), (1, ))
    assert_size_stride(primals_209, (512, ), (1, ))
    assert_size_stride(primals_210, (512, ), (1, ))
    assert_size_stride(primals_211, (128, ), (1, ))
    assert_size_stride(primals_212, (128, ), (1, ))
    assert_size_stride(primals_213, (128, ), (1, ))
    assert_size_stride(primals_215, (128, ), (1, ))
    assert_size_stride(primals_216, (128, ), (1, ))
    assert_size_stride(primals_217, (128, ), (1, ))
    assert_size_stride(primals_218, (128, ), (1, ))
    assert_size_stride(primals_219, (128, ), (1, ))
    assert_size_stride(primals_220, (128, ), (1, ))
    assert_size_stride(primals_221, (128, ), (1, ))
    assert_size_stride(primals_222, (128, ), (1, ))
    assert_size_stride(primals_223, (128, ), (1, ))
    assert_size_stride(primals_225, (512, ), (1, ))
    assert_size_stride(primals_226, (512, ), (1, ))
    assert_size_stride(primals_227, (128, ), (1, ))
    assert_size_stride(primals_228, (128, ), (1, ))
    assert_size_stride(primals_229, (128, ), (1, ))
    assert_size_stride(primals_231, (128, ), (1, ))
    assert_size_stride(primals_232, (128, ), (1, ))
    assert_size_stride(primals_233, (128, ), (1, ))
    assert_size_stride(primals_234, (128, ), (1, ))
    assert_size_stride(primals_235, (128, ), (1, ))
    assert_size_stride(primals_236, (128, ), (1, ))
    assert_size_stride(primals_237, (128, ), (1, ))
    assert_size_stride(primals_238, (128, ), (1, ))
    assert_size_stride(primals_239, (128, ), (1, ))
    assert_size_stride(primals_241, (512, ), (1, ))
    assert_size_stride(primals_242, (512, ), (1, ))
    assert_size_stride(primals_243, (128, ), (1, ))
    assert_size_stride(primals_244, (128, ), (1, ))
    assert_size_stride(primals_245, (128, ), (1, ))
    assert_size_stride(primals_247, (128, ), (1, ))
    assert_size_stride(primals_248, (128, ), (1, ))
    assert_size_stride(primals_249, (128, ), (1, ))
    assert_size_stride(primals_250, (128, ), (1, ))
    assert_size_stride(primals_251, (128, ), (1, ))
    assert_size_stride(primals_252, (128, ), (1, ))
    assert_size_stride(primals_253, (128, ), (1, ))
    assert_size_stride(primals_254, (128, ), (1, ))
    assert_size_stride(primals_255, (128, ), (1, ))
    assert_size_stride(primals_257, (512, ), (1, ))
    assert_size_stride(primals_258, (512, ), (1, ))
    assert_size_stride(primals_259, (128, ), (1, ))
    assert_size_stride(primals_260, (128, ), (1, ))
    assert_size_stride(primals_261, (128, ), (1, ))
    assert_size_stride(primals_263, (128, ), (1, ))
    assert_size_stride(primals_264, (128, ), (1, ))
    assert_size_stride(primals_265, (128, ), (1, ))
    assert_size_stride(primals_266, (128, ), (1, ))
    assert_size_stride(primals_267, (128, ), (1, ))
    assert_size_stride(primals_268, (128, ), (1, ))
    assert_size_stride(primals_269, (128, ), (1, ))
    assert_size_stride(primals_270, (128, ), (1, ))
    assert_size_stride(primals_271, (128, ), (1, ))
    assert_size_stride(primals_273, (512, ), (1, ))
    assert_size_stride(primals_274, (512, ), (1, ))
    assert_size_stride(primals_275, (128, ), (1, ))
    assert_size_stride(primals_276, (128, ), (1, ))
    assert_size_stride(primals_277, (128, ), (1, ))
    assert_size_stride(primals_279, (128, ), (1, ))
    assert_size_stride(primals_280, (128, ), (1, ))
    assert_size_stride(primals_281, (128, ), (1, ))
    assert_size_stride(primals_282, (128, ), (1, ))
    assert_size_stride(primals_283, (128, ), (1, ))
    assert_size_stride(primals_284, (128, ), (1, ))
    assert_size_stride(primals_285, (128, ), (1, ))
    assert_size_stride(primals_286, (128, ), (1, ))
    assert_size_stride(primals_287, (128, ), (1, ))
    assert_size_stride(primals_289, (512, ), (1, ))
    assert_size_stride(primals_290, (512, ), (1, ))
    assert_size_stride(primals_291, (128, ), (1, ))
    assert_size_stride(primals_292, (128, ), (1, ))
    assert_size_stride(primals_293, (128, ), (1, ))
    assert_size_stride(primals_295, (128, ), (1, ))
    assert_size_stride(primals_296, (128, ), (1, ))
    assert_size_stride(primals_297, (128, ), (1, ))
    assert_size_stride(primals_298, (128, ), (1, ))
    assert_size_stride(primals_299, (128, ), (1, ))
    assert_size_stride(primals_300, (128, ), (1, ))
    assert_size_stride(primals_301, (128, ), (1, ))
    assert_size_stride(primals_302, (128, ), (1, ))
    assert_size_stride(primals_303, (128, ), (1, ))
    assert_size_stride(primals_305, (512, ), (1, ))
    assert_size_stride(primals_306, (512, ), (1, ))
    assert_size_stride(primals_307, (128, ), (1, ))
    assert_size_stride(primals_308, (128, ), (1, ))
    assert_size_stride(primals_309, (128, ), (1, ))
    assert_size_stride(primals_311, (128, ), (1, ))
    assert_size_stride(primals_312, (128, ), (1, ))
    assert_size_stride(primals_313, (128, ), (1, ))
    assert_size_stride(primals_314, (128, ), (1, ))
    assert_size_stride(primals_315, (128, ), (1, ))
    assert_size_stride(primals_316, (128, ), (1, ))
    assert_size_stride(primals_317, (128, ), (1, ))
    assert_size_stride(primals_318, (128, ), (1, ))
    assert_size_stride(primals_319, (128, ), (1, ))
    assert_size_stride(primals_321, (512, ), (1, ))
    assert_size_stride(primals_322, (512, ), (1, ))
    assert_size_stride(primals_323, (128, ), (1, ))
    assert_size_stride(primals_324, (128, ), (1, ))
    assert_size_stride(primals_325, (128, ), (1, ))
    assert_size_stride(primals_327, (128, ), (1, ))
    assert_size_stride(primals_328, (128, ), (1, ))
    assert_size_stride(primals_329, (128, ), (1, ))
    assert_size_stride(primals_330, (128, ), (1, ))
    assert_size_stride(primals_331, (128, ), (1, ))
    assert_size_stride(primals_332, (128, ), (1, ))
    assert_size_stride(primals_333, (128, ), (1, ))
    assert_size_stride(primals_334, (128, ), (1, ))
    assert_size_stride(primals_335, (128, ), (1, ))
    assert_size_stride(primals_337, (512, ), (1, ))
    assert_size_stride(primals_338, (512, ), (1, ))
    assert_size_stride(primals_339, (128, ), (1, ))
    assert_size_stride(primals_340, (128, ), (1, ))
    assert_size_stride(primals_341, (128, ), (1, ))
    assert_size_stride(primals_343, (128, ), (1, ))
    assert_size_stride(primals_344, (128, ), (1, ))
    assert_size_stride(primals_345, (128, ), (1, ))
    assert_size_stride(primals_346, (128, ), (1, ))
    assert_size_stride(primals_347, (128, ), (1, ))
    assert_size_stride(primals_348, (128, ), (1, ))
    assert_size_stride(primals_349, (128, ), (1, ))
    assert_size_stride(primals_350, (128, ), (1, ))
    assert_size_stride(primals_351, (128, ), (1, ))
    assert_size_stride(primals_353, (512, ), (1, ))
    assert_size_stride(primals_354, (512, ), (1, ))
    assert_size_stride(primals_355, (128, ), (1, ))
    assert_size_stride(primals_356, (128, ), (1, ))
    assert_size_stride(primals_357, (128, ), (1, ))
    assert_size_stride(primals_359, (128, ), (1, ))
    assert_size_stride(primals_360, (128, ), (1, ))
    assert_size_stride(primals_361, (128, ), (1, ))
    assert_size_stride(primals_362, (128, ), (1, ))
    assert_size_stride(primals_363, (128, ), (1, ))
    assert_size_stride(primals_364, (128, ), (1, ))
    assert_size_stride(primals_365, (128, ), (1, ))
    assert_size_stride(primals_366, (128, ), (1, ))
    assert_size_stride(primals_367, (128, ), (1, ))
    assert_size_stride(primals_369, (512, ), (1, ))
    assert_size_stride(primals_370, (512, ), (1, ))
    assert_size_stride(primals_371, (128, ), (1, ))
    assert_size_stride(primals_372, (128, ), (1, ))
    assert_size_stride(primals_373, (128, ), (1, ))
    assert_size_stride(primals_375, (128, ), (1, ))
    assert_size_stride(primals_376, (128, ), (1, ))
    assert_size_stride(primals_377, (128, ), (1, ))
    assert_size_stride(primals_378, (128, ), (1, ))
    assert_size_stride(primals_379, (128, ), (1, ))
    assert_size_stride(primals_380, (128, ), (1, ))
    assert_size_stride(primals_381, (128, ), (1, ))
    assert_size_stride(primals_382, (128, ), (1, ))
    assert_size_stride(primals_383, (128, ), (1, ))
    assert_size_stride(primals_385, (512, ), (1, ))
    assert_size_stride(primals_1117, (512, ), (1, ))
    assert_size_stride(primals_1120, (1, 128), (128, 1))
    assert_size_stride(primals_1121, (1, 128), (128, 1))
    assert_size_stride(full_default, (1, 128), (128, 1))
    assert_size_stride(slice_4, (1, 128), (512, 1))
    assert_size_stride(view, (128, 384), (384, 1))
    assert_size_stride(add_1, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(view_2, (128, 512), (512, 1))
    assert_size_stride(addmm_1, (128, 128), (128, 1))
    assert_size_stride(addmm_2, (128, 128), (128, 1))
    assert_size_stride(view_6, (128, 128), (128, 1))
    assert_size_stride(clone_default_69, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_70, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_71, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_212, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_213, (), ())
    assert_size_stride(getitem_214, (), ())
    assert_size_stride(alias_default_47, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_22, (128, 128), (128, 1))
    assert_size_stride(addmm_6, (128, 128), (128, 1))
    assert_size_stride(view_24, (128, 128), (128, 1))
    assert_size_stride(view_26, (128, 512), (512, 1))
    assert_size_stride(addmm_8, (128, 128), (128, 1))
    assert_size_stride(view_28, (128, 128), (128, 1))
    assert_size_stride(view_30, (128, 512), (512, 1))
    assert_size_stride(addmm_10, (128, 128), (128, 1))
    assert_size_stride(view_32, (128, 128), (128, 1))
    assert_size_stride(view_34, (128, 512), (512, 1))
    assert_size_stride(addmm_12, (128, 128), (128, 1))
    assert_size_stride(view_36, (128, 128), (128, 1))
    assert_size_stride(view_38, (128, 512), (512, 1))
    assert_size_stride(addmm_14, (128, 128), (128, 1))
    assert_size_stride(view_40, (128, 128), (128, 1))
    assert_size_stride(add_16, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(view_42, (128, 512), (512, 1))
    assert_size_stride(addmm_16, (128, 128), (128, 1))
    assert_size_stride(addmm_17, (128, 128), (128, 1))
    assert_size_stride(view_46, (128, 128), (128, 1))
    assert_size_stride(clone_default_66, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_67, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_68, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_205, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_206, (), ())
    assert_size_stride(getitem_207, (), ())
    assert_size_stride(alias_default_45, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_62, (128, 128), (128, 1))
    assert_size_stride(addmm_21, (128, 128), (128, 1))
    assert_size_stride(view_64, (128, 128), (128, 1))
    assert_size_stride(view_66, (128, 512), (512, 1))
    assert_size_stride(addmm_23, (128, 128), (128, 1))
    assert_size_stride(view_68, (128, 128), (128, 1))
    assert_size_stride(view_70, (128, 512), (512, 1))
    assert_size_stride(addmm_25, (128, 128), (128, 1))
    assert_size_stride(view_72, (128, 128), (128, 1))
    assert_size_stride(view_74, (128, 512), (512, 1))
    assert_size_stride(addmm_27, (128, 128), (128, 1))
    assert_size_stride(view_76, (128, 128), (128, 1))
    assert_size_stride(view_78, (128, 512), (512, 1))
    assert_size_stride(addmm_29, (128, 128), (128, 1))
    assert_size_stride(view_80, (128, 128), (128, 1))
    assert_size_stride(addmm_30, (128, 512), (512, 1))
    assert_size_stride(view_82, (128, 512), (512, 1))
    assert_size_stride(addmm_31, (128, 128), (128, 1))
    assert_size_stride(addmm_32, (128, 128), (128, 1))
    assert_size_stride(view_86, (128, 128), (128, 1))
    assert_size_stride(clone_default_63, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_64, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_65, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_198, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_199, (), ())
    assert_size_stride(getitem_200, (), ())
    assert_size_stride(alias_default_43, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_102, (128, 128), (128, 1))
    assert_size_stride(addmm_36, (128, 128), (128, 1))
    assert_size_stride(view_104, (128, 128), (128, 1))
    assert_size_stride(view_106, (128, 512), (512, 1))
    assert_size_stride(addmm_38, (128, 128), (128, 1))
    assert_size_stride(view_108, (128, 128), (128, 1))
    assert_size_stride(view_110, (128, 512), (512, 1))
    assert_size_stride(addmm_40, (128, 128), (128, 1))
    assert_size_stride(view_112, (128, 128), (128, 1))
    assert_size_stride(view_114, (128, 512), (512, 1))
    assert_size_stride(addmm_42, (128, 128), (128, 1))
    assert_size_stride(view_116, (128, 128), (128, 1))
    assert_size_stride(view_118, (128, 512), (512, 1))
    assert_size_stride(addmm_44, (128, 128), (128, 1))
    assert_size_stride(view_120, (128, 128), (128, 1))
    assert_size_stride(addmm_45, (128, 512), (512, 1))
    assert_size_stride(view_122, (128, 512), (512, 1))
    assert_size_stride(addmm_46, (128, 128), (128, 1))
    assert_size_stride(addmm_47, (128, 128), (128, 1))
    assert_size_stride(view_126, (128, 128), (128, 1))
    assert_size_stride(clone_default_60, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_61, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_62, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_191, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_192, (), ())
    assert_size_stride(getitem_193, (), ())
    assert_size_stride(alias_default_41, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_142, (128, 128), (128, 1))
    assert_size_stride(addmm_51, (128, 128), (128, 1))
    assert_size_stride(view_144, (128, 128), (128, 1))
    assert_size_stride(view_146, (128, 512), (512, 1))
    assert_size_stride(addmm_53, (128, 128), (128, 1))
    assert_size_stride(view_148, (128, 128), (128, 1))
    assert_size_stride(view_150, (128, 512), (512, 1))
    assert_size_stride(addmm_55, (128, 128), (128, 1))
    assert_size_stride(view_152, (128, 128), (128, 1))
    assert_size_stride(view_154, (128, 512), (512, 1))
    assert_size_stride(addmm_57, (128, 128), (128, 1))
    assert_size_stride(view_156, (128, 128), (128, 1))
    assert_size_stride(view_158, (128, 512), (512, 1))
    assert_size_stride(addmm_59, (128, 128), (128, 1))
    assert_size_stride(view_160, (128, 128), (128, 1))
    assert_size_stride(addmm_60, (128, 512), (512, 1))
    assert_size_stride(view_162, (128, 512), (512, 1))
    assert_size_stride(addmm_61, (128, 128), (128, 1))
    assert_size_stride(addmm_62, (128, 128), (128, 1))
    assert_size_stride(view_166, (128, 128), (128, 1))
    assert_size_stride(clone_default_57, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_58, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_59, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_184, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_185, (), ())
    assert_size_stride(getitem_186, (), ())
    assert_size_stride(alias_default_39, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_182, (128, 128), (128, 1))
    assert_size_stride(addmm_66, (128, 128), (128, 1))
    assert_size_stride(view_184, (128, 128), (128, 1))
    assert_size_stride(view_186, (128, 512), (512, 1))
    assert_size_stride(addmm_68, (128, 128), (128, 1))
    assert_size_stride(view_188, (128, 128), (128, 1))
    assert_size_stride(view_190, (128, 512), (512, 1))
    assert_size_stride(addmm_70, (128, 128), (128, 1))
    assert_size_stride(view_192, (128, 128), (128, 1))
    assert_size_stride(view_194, (128, 512), (512, 1))
    assert_size_stride(addmm_72, (128, 128), (128, 1))
    assert_size_stride(view_196, (128, 128), (128, 1))
    assert_size_stride(view_198, (128, 512), (512, 1))
    assert_size_stride(addmm_74, (128, 128), (128, 1))
    assert_size_stride(view_200, (128, 128), (128, 1))
    assert_size_stride(addmm_75, (128, 512), (512, 1))
    assert_size_stride(view_202, (128, 512), (512, 1))
    assert_size_stride(addmm_76, (128, 128), (128, 1))
    assert_size_stride(addmm_77, (128, 128), (128, 1))
    assert_size_stride(view_206, (128, 128), (128, 1))
    assert_size_stride(clone_default_54, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_55, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_56, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_177, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_178, (), ())
    assert_size_stride(getitem_179, (), ())
    assert_size_stride(alias_default_37, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_222, (128, 128), (128, 1))
    assert_size_stride(addmm_81, (128, 128), (128, 1))
    assert_size_stride(view_224, (128, 128), (128, 1))
    assert_size_stride(view_226, (128, 512), (512, 1))
    assert_size_stride(addmm_83, (128, 128), (128, 1))
    assert_size_stride(view_228, (128, 128), (128, 1))
    assert_size_stride(view_230, (128, 512), (512, 1))
    assert_size_stride(addmm_85, (128, 128), (128, 1))
    assert_size_stride(view_232, (128, 128), (128, 1))
    assert_size_stride(view_234, (128, 512), (512, 1))
    assert_size_stride(addmm_87, (128, 128), (128, 1))
    assert_size_stride(view_236, (128, 128), (128, 1))
    assert_size_stride(view_238, (128, 512), (512, 1))
    assert_size_stride(addmm_89, (128, 128), (128, 1))
    assert_size_stride(view_240, (128, 128), (128, 1))
    assert_size_stride(addmm_90, (128, 512), (512, 1))
    assert_size_stride(view_242, (128, 512), (512, 1))
    assert_size_stride(addmm_91, (128, 128), (128, 1))
    assert_size_stride(addmm_92, (128, 128), (128, 1))
    assert_size_stride(view_246, (128, 128), (128, 1))
    assert_size_stride(clone_default_51, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_52, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_53, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_170, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_171, (), ())
    assert_size_stride(getitem_172, (), ())
    assert_size_stride(alias_default_35, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_262, (128, 128), (128, 1))
    assert_size_stride(addmm_96, (128, 128), (128, 1))
    assert_size_stride(view_264, (128, 128), (128, 1))
    assert_size_stride(view_266, (128, 512), (512, 1))
    assert_size_stride(addmm_98, (128, 128), (128, 1))
    assert_size_stride(view_268, (128, 128), (128, 1))
    assert_size_stride(view_270, (128, 512), (512, 1))
    assert_size_stride(addmm_100, (128, 128), (128, 1))
    assert_size_stride(view_272, (128, 128), (128, 1))
    assert_size_stride(view_274, (128, 512), (512, 1))
    assert_size_stride(addmm_102, (128, 128), (128, 1))
    assert_size_stride(view_276, (128, 128), (128, 1))
    assert_size_stride(view_278, (128, 512), (512, 1))
    assert_size_stride(addmm_104, (128, 128), (128, 1))
    assert_size_stride(view_280, (128, 128), (128, 1))
    assert_size_stride(addmm_105, (128, 512), (512, 1))
    assert_size_stride(view_282, (128, 512), (512, 1))
    assert_size_stride(addmm_106, (128, 128), (128, 1))
    assert_size_stride(addmm_107, (128, 128), (128, 1))
    assert_size_stride(view_286, (128, 128), (128, 1))
    assert_size_stride(clone_default_48, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_49, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_50, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_163, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_164, (), ())
    assert_size_stride(getitem_165, (), ())
    assert_size_stride(alias_default_33, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_302, (128, 128), (128, 1))
    assert_size_stride(addmm_111, (128, 128), (128, 1))
    assert_size_stride(view_304, (128, 128), (128, 1))
    assert_size_stride(view_306, (128, 512), (512, 1))
    assert_size_stride(addmm_113, (128, 128), (128, 1))
    assert_size_stride(view_308, (128, 128), (128, 1))
    assert_size_stride(view_310, (128, 512), (512, 1))
    assert_size_stride(addmm_115, (128, 128), (128, 1))
    assert_size_stride(view_312, (128, 128), (128, 1))
    assert_size_stride(view_314, (128, 512), (512, 1))
    assert_size_stride(addmm_117, (128, 128), (128, 1))
    assert_size_stride(view_316, (128, 128), (128, 1))
    assert_size_stride(view_318, (128, 512), (512, 1))
    assert_size_stride(addmm_119, (128, 128), (128, 1))
    assert_size_stride(view_320, (128, 128), (128, 1))
    assert_size_stride(addmm_120, (128, 512), (512, 1))
    assert_size_stride(view_322, (128, 512), (512, 1))
    assert_size_stride(addmm_121, (128, 128), (128, 1))
    assert_size_stride(addmm_122, (128, 128), (128, 1))
    assert_size_stride(view_326, (128, 128), (128, 1))
    assert_size_stride(clone_default_45, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_46, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_47, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_156, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_157, (), ())
    assert_size_stride(getitem_158, (), ())
    assert_size_stride(alias_default_31, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_342, (128, 128), (128, 1))
    assert_size_stride(addmm_126, (128, 128), (128, 1))
    assert_size_stride(view_344, (128, 128), (128, 1))
    assert_size_stride(view_346, (128, 512), (512, 1))
    assert_size_stride(addmm_128, (128, 128), (128, 1))
    assert_size_stride(view_348, (128, 128), (128, 1))
    assert_size_stride(view_350, (128, 512), (512, 1))
    assert_size_stride(addmm_130, (128, 128), (128, 1))
    assert_size_stride(view_352, (128, 128), (128, 1))
    assert_size_stride(view_354, (128, 512), (512, 1))
    assert_size_stride(addmm_132, (128, 128), (128, 1))
    assert_size_stride(view_356, (128, 128), (128, 1))
    assert_size_stride(view_358, (128, 512), (512, 1))
    assert_size_stride(addmm_134, (128, 128), (128, 1))
    assert_size_stride(view_360, (128, 128), (128, 1))
    assert_size_stride(addmm_135, (128, 512), (512, 1))
    assert_size_stride(view_362, (128, 512), (512, 1))
    assert_size_stride(addmm_136, (128, 128), (128, 1))
    assert_size_stride(addmm_137, (128, 128), (128, 1))
    assert_size_stride(view_366, (128, 128), (128, 1))
    assert_size_stride(clone_default_42, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_43, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_44, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_149, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_150, (), ())
    assert_size_stride(getitem_151, (), ())
    assert_size_stride(alias_default_29, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_382, (128, 128), (128, 1))
    assert_size_stride(addmm_141, (128, 128), (128, 1))
    assert_size_stride(view_384, (128, 128), (128, 1))
    assert_size_stride(view_386, (128, 512), (512, 1))
    assert_size_stride(addmm_143, (128, 128), (128, 1))
    assert_size_stride(view_388, (128, 128), (128, 1))
    assert_size_stride(view_390, (128, 512), (512, 1))
    assert_size_stride(addmm_145, (128, 128), (128, 1))
    assert_size_stride(view_392, (128, 128), (128, 1))
    assert_size_stride(view_394, (128, 512), (512, 1))
    assert_size_stride(addmm_147, (128, 128), (128, 1))
    assert_size_stride(view_396, (128, 128), (128, 1))
    assert_size_stride(view_398, (128, 512), (512, 1))
    assert_size_stride(addmm_149, (128, 128), (128, 1))
    assert_size_stride(view_400, (128, 128), (128, 1))
    assert_size_stride(addmm_150, (128, 512), (512, 1))
    assert_size_stride(view_402, (128, 512), (512, 1))
    assert_size_stride(addmm_151, (128, 128), (128, 1))
    assert_size_stride(addmm_152, (128, 128), (128, 1))
    assert_size_stride(view_406, (128, 128), (128, 1))
    assert_size_stride(clone_default_39, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_40, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_41, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_142, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_143, (), ())
    assert_size_stride(getitem_144, (), ())
    assert_size_stride(alias_default_27, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_422, (128, 128), (128, 1))
    assert_size_stride(addmm_156, (128, 128), (128, 1))
    assert_size_stride(view_424, (128, 128), (128, 1))
    assert_size_stride(view_426, (128, 512), (512, 1))
    assert_size_stride(addmm_158, (128, 128), (128, 1))
    assert_size_stride(view_428, (128, 128), (128, 1))
    assert_size_stride(view_430, (128, 512), (512, 1))
    assert_size_stride(addmm_160, (128, 128), (128, 1))
    assert_size_stride(view_432, (128, 128), (128, 1))
    assert_size_stride(view_434, (128, 512), (512, 1))
    assert_size_stride(addmm_162, (128, 128), (128, 1))
    assert_size_stride(view_436, (128, 128), (128, 1))
    assert_size_stride(view_438, (128, 512), (512, 1))
    assert_size_stride(addmm_164, (128, 128), (128, 1))
    assert_size_stride(view_440, (128, 128), (128, 1))
    assert_size_stride(addmm_165, (128, 512), (512, 1))
    assert_size_stride(view_442, (128, 512), (512, 1))
    assert_size_stride(addmm_166, (128, 128), (128, 1))
    assert_size_stride(addmm_167, (128, 128), (128, 1))
    assert_size_stride(view_446, (128, 128), (128, 1))
    assert_size_stride(clone_default_36, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_37, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_38, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_135, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_136, (), ())
    assert_size_stride(getitem_137, (), ())
    assert_size_stride(alias_default_25, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_462, (128, 128), (128, 1))
    assert_size_stride(addmm_171, (128, 128), (128, 1))
    assert_size_stride(view_464, (128, 128), (128, 1))
    assert_size_stride(view_466, (128, 512), (512, 1))
    assert_size_stride(addmm_173, (128, 128), (128, 1))
    assert_size_stride(view_468, (128, 128), (128, 1))
    assert_size_stride(view_470, (128, 512), (512, 1))
    assert_size_stride(addmm_175, (128, 128), (128, 1))
    assert_size_stride(view_472, (128, 128), (128, 1))
    assert_size_stride(view_474, (128, 512), (512, 1))
    assert_size_stride(addmm_177, (128, 128), (128, 1))
    assert_size_stride(view_476, (128, 128), (128, 1))
    assert_size_stride(view_478, (128, 512), (512, 1))
    assert_size_stride(addmm_179, (128, 128), (128, 1))
    assert_size_stride(view_480, (128, 128), (128, 1))
    assert_size_stride(addmm_180, (128, 512), (512, 1))
    assert_size_stride(view_482, (128, 512), (512, 1))
    assert_size_stride(addmm_181, (128, 128), (128, 1))
    assert_size_stride(addmm_182, (128, 128), (128, 1))
    assert_size_stride(view_486, (128, 128), (128, 1))
    assert_size_stride(clone_default_33, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_34, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_35, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_128, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_129, (), ())
    assert_size_stride(getitem_130, (), ())
    assert_size_stride(alias_default_23, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_502, (128, 128), (128, 1))
    assert_size_stride(addmm_186, (128, 128), (128, 1))
    assert_size_stride(view_504, (128, 128), (128, 1))
    assert_size_stride(view_506, (128, 512), (512, 1))
    assert_size_stride(addmm_188, (128, 128), (128, 1))
    assert_size_stride(view_508, (128, 128), (128, 1))
    assert_size_stride(view_510, (128, 512), (512, 1))
    assert_size_stride(addmm_190, (128, 128), (128, 1))
    assert_size_stride(view_512, (128, 128), (128, 1))
    assert_size_stride(view_514, (128, 512), (512, 1))
    assert_size_stride(addmm_192, (128, 128), (128, 1))
    assert_size_stride(view_516, (128, 128), (128, 1))
    assert_size_stride(view_518, (128, 512), (512, 1))
    assert_size_stride(addmm_194, (128, 128), (128, 1))
    assert_size_stride(view_520, (128, 128), (128, 1))
    assert_size_stride(addmm_195, (128, 512), (512, 1))
    assert_size_stride(view_522, (128, 512), (512, 1))
    assert_size_stride(addmm_196, (128, 128), (128, 1))
    assert_size_stride(addmm_197, (128, 128), (128, 1))
    assert_size_stride(view_526, (128, 128), (128, 1))
    assert_size_stride(clone_default_30, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_31, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_32, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_121, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_122, (), ())
    assert_size_stride(getitem_123, (), ())
    assert_size_stride(alias_default_21, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_542, (128, 128), (128, 1))
    assert_size_stride(addmm_201, (128, 128), (128, 1))
    assert_size_stride(view_544, (128, 128), (128, 1))
    assert_size_stride(view_546, (128, 512), (512, 1))
    assert_size_stride(addmm_203, (128, 128), (128, 1))
    assert_size_stride(view_548, (128, 128), (128, 1))
    assert_size_stride(view_550, (128, 512), (512, 1))
    assert_size_stride(addmm_205, (128, 128), (128, 1))
    assert_size_stride(view_552, (128, 128), (128, 1))
    assert_size_stride(view_554, (128, 512), (512, 1))
    assert_size_stride(addmm_207, (128, 128), (128, 1))
    assert_size_stride(view_556, (128, 128), (128, 1))
    assert_size_stride(view_558, (128, 512), (512, 1))
    assert_size_stride(addmm_209, (128, 128), (128, 1))
    assert_size_stride(view_560, (128, 128), (128, 1))
    assert_size_stride(addmm_210, (128, 512), (512, 1))
    assert_size_stride(view_562, (128, 512), (512, 1))
    assert_size_stride(addmm_211, (128, 128), (128, 1))
    assert_size_stride(addmm_212, (128, 128), (128, 1))
    assert_size_stride(view_566, (128, 128), (128, 1))
    assert_size_stride(clone_default_27, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_28, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_29, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_114, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_115, (), ())
    assert_size_stride(getitem_116, (), ())
    assert_size_stride(alias_default_19, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_582, (128, 128), (128, 1))
    assert_size_stride(addmm_216, (128, 128), (128, 1))
    assert_size_stride(view_584, (128, 128), (128, 1))
    assert_size_stride(view_586, (128, 512), (512, 1))
    assert_size_stride(addmm_218, (128, 128), (128, 1))
    assert_size_stride(view_588, (128, 128), (128, 1))
    assert_size_stride(view_590, (128, 512), (512, 1))
    assert_size_stride(addmm_220, (128, 128), (128, 1))
    assert_size_stride(view_592, (128, 128), (128, 1))
    assert_size_stride(view_594, (128, 512), (512, 1))
    assert_size_stride(addmm_222, (128, 128), (128, 1))
    assert_size_stride(view_596, (128, 128), (128, 1))
    assert_size_stride(view_598, (128, 512), (512, 1))
    assert_size_stride(addmm_224, (128, 128), (128, 1))
    assert_size_stride(view_600, (128, 128), (128, 1))
    assert_size_stride(addmm_225, (128, 512), (512, 1))
    assert_size_stride(view_602, (128, 512), (512, 1))
    assert_size_stride(addmm_226, (128, 128), (128, 1))
    assert_size_stride(addmm_227, (128, 128), (128, 1))
    assert_size_stride(view_606, (128, 128), (128, 1))
    assert_size_stride(clone_default_24, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_25, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_26, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_107, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_108, (), ())
    assert_size_stride(getitem_109, (), ())
    assert_size_stride(alias_default_17, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_622, (128, 128), (128, 1))
    assert_size_stride(addmm_231, (128, 128), (128, 1))
    assert_size_stride(view_624, (128, 128), (128, 1))
    assert_size_stride(view_626, (128, 512), (512, 1))
    assert_size_stride(addmm_233, (128, 128), (128, 1))
    assert_size_stride(view_628, (128, 128), (128, 1))
    assert_size_stride(view_630, (128, 512), (512, 1))
    assert_size_stride(addmm_235, (128, 128), (128, 1))
    assert_size_stride(view_632, (128, 128), (128, 1))
    assert_size_stride(view_634, (128, 512), (512, 1))
    assert_size_stride(addmm_237, (128, 128), (128, 1))
    assert_size_stride(view_636, (128, 128), (128, 1))
    assert_size_stride(view_638, (128, 512), (512, 1))
    assert_size_stride(addmm_239, (128, 128), (128, 1))
    assert_size_stride(view_640, (128, 128), (128, 1))
    assert_size_stride(addmm_240, (128, 512), (512, 1))
    assert_size_stride(view_642, (128, 512), (512, 1))
    assert_size_stride(addmm_241, (128, 128), (128, 1))
    assert_size_stride(addmm_242, (128, 128), (128, 1))
    assert_size_stride(view_646, (128, 128), (128, 1))
    assert_size_stride(clone_default_21, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_22, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_23, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_100, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_101, (), ())
    assert_size_stride(getitem_102, (), ())
    assert_size_stride(alias_default_15, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_662, (128, 128), (128, 1))
    assert_size_stride(addmm_246, (128, 128), (128, 1))
    assert_size_stride(view_664, (128, 128), (128, 1))
    assert_size_stride(view_666, (128, 512), (512, 1))
    assert_size_stride(addmm_248, (128, 128), (128, 1))
    assert_size_stride(view_668, (128, 128), (128, 1))
    assert_size_stride(view_670, (128, 512), (512, 1))
    assert_size_stride(addmm_250, (128, 128), (128, 1))
    assert_size_stride(view_672, (128, 128), (128, 1))
    assert_size_stride(view_674, (128, 512), (512, 1))
    assert_size_stride(addmm_252, (128, 128), (128, 1))
    assert_size_stride(view_676, (128, 128), (128, 1))
    assert_size_stride(view_678, (128, 512), (512, 1))
    assert_size_stride(addmm_254, (128, 128), (128, 1))
    assert_size_stride(view_680, (128, 128), (128, 1))
    assert_size_stride(addmm_255, (128, 512), (512, 1))
    assert_size_stride(view_682, (128, 512), (512, 1))
    assert_size_stride(addmm_256, (128, 128), (128, 1))
    assert_size_stride(addmm_257, (128, 128), (128, 1))
    assert_size_stride(view_686, (128, 128), (128, 1))
    assert_size_stride(clone_default_18, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_19, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_20, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_93, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_94, (), ())
    assert_size_stride(getitem_95, (), ())
    assert_size_stride(alias_default_13, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_702, (128, 128), (128, 1))
    assert_size_stride(addmm_261, (128, 128), (128, 1))
    assert_size_stride(view_704, (128, 128), (128, 1))
    assert_size_stride(view_706, (128, 512), (512, 1))
    assert_size_stride(addmm_263, (128, 128), (128, 1))
    assert_size_stride(view_708, (128, 128), (128, 1))
    assert_size_stride(view_710, (128, 512), (512, 1))
    assert_size_stride(addmm_265, (128, 128), (128, 1))
    assert_size_stride(view_712, (128, 128), (128, 1))
    assert_size_stride(view_714, (128, 512), (512, 1))
    assert_size_stride(addmm_267, (128, 128), (128, 1))
    assert_size_stride(view_716, (128, 128), (128, 1))
    assert_size_stride(view_718, (128, 512), (512, 1))
    assert_size_stride(addmm_269, (128, 128), (128, 1))
    assert_size_stride(view_720, (128, 128), (128, 1))
    assert_size_stride(addmm_270, (128, 512), (512, 1))
    assert_size_stride(view_722, (128, 512), (512, 1))
    assert_size_stride(addmm_271, (128, 128), (128, 1))
    assert_size_stride(addmm_272, (128, 128), (128, 1))
    assert_size_stride(view_726, (128, 128), (128, 1))
    assert_size_stride(clone_default_15, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_16, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_17, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_86, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_87, (), ())
    assert_size_stride(getitem_88, (), ())
    assert_size_stride(alias_default_11, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_742, (128, 128), (128, 1))
    assert_size_stride(addmm_276, (128, 128), (128, 1))
    assert_size_stride(view_744, (128, 128), (128, 1))
    assert_size_stride(view_746, (128, 512), (512, 1))
    assert_size_stride(addmm_278, (128, 128), (128, 1))
    assert_size_stride(view_748, (128, 128), (128, 1))
    assert_size_stride(view_750, (128, 512), (512, 1))
    assert_size_stride(addmm_280, (128, 128), (128, 1))
    assert_size_stride(view_752, (128, 128), (128, 1))
    assert_size_stride(view_754, (128, 512), (512, 1))
    assert_size_stride(addmm_282, (128, 128), (128, 1))
    assert_size_stride(view_756, (128, 128), (128, 1))
    assert_size_stride(view_758, (128, 512), (512, 1))
    assert_size_stride(addmm_284, (128, 128), (128, 1))
    assert_size_stride(view_760, (128, 128), (128, 1))
    assert_size_stride(addmm_285, (128, 512), (512, 1))
    assert_size_stride(view_762, (128, 512), (512, 1))
    assert_size_stride(addmm_286, (128, 128), (128, 1))
    assert_size_stride(addmm_287, (128, 128), (128, 1))
    assert_size_stride(view_766, (128, 128), (128, 1))
    assert_size_stride(clone_default_12, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_13, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_14, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_79, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_80, (), ())
    assert_size_stride(getitem_81, (), ())
    assert_size_stride(alias_default_9, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_782, (128, 128), (128, 1))
    assert_size_stride(addmm_291, (128, 128), (128, 1))
    assert_size_stride(view_784, (128, 128), (128, 1))
    assert_size_stride(view_786, (128, 512), (512, 1))
    assert_size_stride(addmm_293, (128, 128), (128, 1))
    assert_size_stride(view_788, (128, 128), (128, 1))
    assert_size_stride(view_790, (128, 512), (512, 1))
    assert_size_stride(addmm_295, (128, 128), (128, 1))
    assert_size_stride(view_792, (128, 128), (128, 1))
    assert_size_stride(view_794, (128, 512), (512, 1))
    assert_size_stride(addmm_297, (128, 128), (128, 1))
    assert_size_stride(view_796, (128, 128), (128, 1))
    assert_size_stride(view_798, (128, 512), (512, 1))
    assert_size_stride(addmm_299, (128, 128), (128, 1))
    assert_size_stride(view_800, (128, 128), (128, 1))
    assert_size_stride(addmm_300, (128, 512), (512, 1))
    assert_size_stride(view_802, (128, 512), (512, 1))
    assert_size_stride(addmm_301, (128, 128), (128, 1))
    assert_size_stride(addmm_302, (128, 128), (128, 1))
    assert_size_stride(view_806, (128, 128), (128, 1))
    assert_size_stride(clone_default_9, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_10, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_11, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_72, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_73, (), ())
    assert_size_stride(getitem_74, (), ())
    assert_size_stride(alias_default_7, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_822, (128, 128), (128, 1))
    assert_size_stride(addmm_306, (128, 128), (128, 1))
    assert_size_stride(view_824, (128, 128), (128, 1))
    assert_size_stride(view_826, (128, 512), (512, 1))
    assert_size_stride(addmm_308, (128, 128), (128, 1))
    assert_size_stride(view_828, (128, 128), (128, 1))
    assert_size_stride(view_830, (128, 512), (512, 1))
    assert_size_stride(addmm_310, (128, 128), (128, 1))
    assert_size_stride(view_832, (128, 128), (128, 1))
    assert_size_stride(view_834, (128, 512), (512, 1))
    assert_size_stride(addmm_312, (128, 128), (128, 1))
    assert_size_stride(view_836, (128, 128), (128, 1))
    assert_size_stride(view_838, (128, 512), (512, 1))
    assert_size_stride(addmm_314, (128, 128), (128, 1))
    assert_size_stride(view_840, (128, 128), (128, 1))
    assert_size_stride(addmm_315, (128, 512), (512, 1))
    assert_size_stride(view_842, (128, 512), (512, 1))
    assert_size_stride(addmm_316, (128, 128), (128, 1))
    assert_size_stride(addmm_317, (128, 128), (128, 1))
    assert_size_stride(view_846, (128, 128), (128, 1))
    assert_size_stride(clone_default_6, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_7, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_8, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_65, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_66, (), ())
    assert_size_stride(getitem_67, (), ())
    assert_size_stride(alias_default_5, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_862, (128, 128), (128, 1))
    assert_size_stride(addmm_321, (128, 128), (128, 1))
    assert_size_stride(view_864, (128, 128), (128, 1))
    assert_size_stride(view_866, (128, 512), (512, 1))
    assert_size_stride(addmm_323, (128, 128), (128, 1))
    assert_size_stride(view_868, (128, 128), (128, 1))
    assert_size_stride(view_870, (128, 512), (512, 1))
    assert_size_stride(addmm_325, (128, 128), (128, 1))
    assert_size_stride(view_872, (128, 128), (128, 1))
    assert_size_stride(view_874, (128, 512), (512, 1))
    assert_size_stride(addmm_327, (128, 128), (128, 1))
    assert_size_stride(view_876, (128, 128), (128, 1))
    assert_size_stride(view_878, (128, 512), (512, 1))
    assert_size_stride(addmm_329, (128, 128), (128, 1))
    assert_size_stride(view_880, (128, 128), (128, 1))
    assert_size_stride(addmm_330, (128, 512), (512, 1))
    assert_size_stride(view_882, (128, 512), (512, 1))
    assert_size_stride(addmm_331, (128, 128), (128, 1))
    assert_size_stride(addmm_332, (128, 128), (128, 1))
    assert_size_stride(view_886, (128, 128), (128, 1))
    assert_size_stride(clone_default_3, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_4, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_5, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_58, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_59, (), ())
    assert_size_stride(getitem_60, (), ())
    assert_size_stride(alias_default_3, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_902, (128, 128), (128, 1))
    assert_size_stride(addmm_336, (128, 128), (128, 1))
    assert_size_stride(view_904, (128, 128), (128, 1))
    assert_size_stride(view_906, (128, 512), (512, 1))
    assert_size_stride(addmm_338, (128, 128), (128, 1))
    assert_size_stride(view_908, (128, 128), (128, 1))
    assert_size_stride(view_910, (128, 512), (512, 1))
    assert_size_stride(addmm_340, (128, 128), (128, 1))
    assert_size_stride(view_912, (128, 128), (128, 1))
    assert_size_stride(view_914, (128, 512), (512, 1))
    assert_size_stride(addmm_342, (128, 128), (128, 1))
    assert_size_stride(view_916, (128, 128), (128, 1))
    assert_size_stride(view_918, (128, 512), (512, 1))
    assert_size_stride(addmm_344, (128, 128), (128, 1))
    assert_size_stride(view_920, (128, 128), (128, 1))
    assert_size_stride(addmm_345, (128, 512), (512, 1))
    assert_size_stride(view_922, (128, 512), (512, 1))
    assert_size_stride(addmm_346, (128, 128), (128, 1))
    assert_size_stride(addmm_347, (128, 128), (128, 1))
    assert_size_stride(view_926, (128, 128), (128, 1))
    assert_size_stride(clone_default, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_1, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(clone_default_2, (1, 4, 128, 32), (16384, 4096, 32, 1))
    assert_size_stride(getitem_51, (1, 4, 128), (512, 128, 1))
    assert_size_stride(getitem_52, (), ())
    assert_size_stride(getitem_53, (), ())
    assert_size_stride(alias_default_1, (1, 4, 128, 32), (16384, 32, 128, 1))
    assert_size_stride(view_942, (128, 128), (128, 1))
    assert_size_stride(addmm_351, (128, 128), (128, 1))
    assert_size_stride(view_944, (128, 128), (128, 1))
    assert_size_stride(view_946, (128, 512), (512, 1))
    assert_size_stride(addmm_353, (128, 128), (128, 1))
    assert_size_stride(view_948, (128, 128), (128, 1))
    assert_size_stride(view_950, (128, 512), (512, 1))
    assert_size_stride(addmm_355, (128, 128), (128, 1))
    assert_size_stride(view_952, (128, 128), (128, 1))
    assert_size_stride(view_954, (128, 512), (512, 1))
    assert_size_stride(addmm_357, (128, 128), (128, 1))
    assert_size_stride(view_956, (128, 128), (128, 1))
    assert_size_stride(view_958, (128, 512), (512, 1))
    assert_size_stride(addmm_359, (128, 128), (128, 1))
    assert_size_stride(view_960, (128, 128), (128, 1))
    assert_size_stride(addmm_360, (128, 512), (512, 1))
    assert_size_stride(view_962, (128, 512), (512, 1))
    assert_size_stride(addmm_361, (128, 512), (512, 1))
    assert_size_stride(getitem_49, (1, 128, 1), (128, 1, 1))
    assert_size_stride(rsqrt, (1, 128, 1), (128, 1, 1))
    assert_size_stride(sub_27, (128, 30522), (30522, 1))
    assert_size_stride(convert_element_type, (), ())
    assert_size_stride(permute_483, (512, 128), (1, 512))
    assert_size_stride(permute_484, (30522, 512), (1, 30522))
    assert_size_stride(permute_486, (512, 512), (512, 1))
    assert_size_stride(permute_490, (512, 128), (128, 1))
    assert_size_stride(permute_494, (128, 512), (512, 1))
    assert_size_stride(le_1, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_498, (512, 128), (128, 1))
    assert_size_stride(permute_502, (128, 512), (512, 1))
    assert_size_stride(le_2, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_506, (512, 128), (128, 1))
    assert_size_stride(permute_510, (128, 512), (512, 1))
    assert_size_stride(le_3, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_514, (512, 128), (128, 1))
    assert_size_stride(permute_518, (128, 512), (512, 1))
    assert_size_stride(le_4, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_522, (512, 128), (128, 1))
    assert_size_stride(permute_526, (128, 128), (128, 1))
    assert_size_stride(permute_539, (128, 512), (512, 1))
    assert_size_stride(permute_543, (128, 128), (128, 1))
    assert_size_stride(permute_547, (128, 128), (128, 1))
    assert_size_stride(permute_551, (128, 512), (512, 1))
    assert_size_stride(permute_555, (128, 512), (512, 1))
    assert_size_stride(permute_559, (512, 128), (128, 1))
    assert_size_stride(permute_563, (128, 512), (512, 1))
    assert_size_stride(le_5, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_567, (512, 128), (128, 1))
    assert_size_stride(permute_571, (128, 512), (512, 1))
    assert_size_stride(le_6, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_575, (512, 128), (128, 1))
    assert_size_stride(permute_579, (128, 512), (512, 1))
    assert_size_stride(le_7, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_583, (512, 128), (128, 1))
    assert_size_stride(permute_587, (128, 512), (512, 1))
    assert_size_stride(le_8, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_591, (512, 128), (128, 1))
    assert_size_stride(permute_595, (128, 128), (128, 1))
    assert_size_stride(permute_608, (128, 512), (512, 1))
    assert_size_stride(permute_612, (128, 128), (128, 1))
    assert_size_stride(permute_616, (128, 128), (128, 1))
    assert_size_stride(permute_620, (128, 512), (512, 1))
    assert_size_stride(permute_624, (128, 512), (512, 1))
    assert_size_stride(permute_628, (512, 128), (128, 1))
    assert_size_stride(permute_632, (128, 512), (512, 1))
    assert_size_stride(le_9, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_636, (512, 128), (128, 1))
    assert_size_stride(permute_640, (128, 512), (512, 1))
    assert_size_stride(le_10, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_644, (512, 128), (128, 1))
    assert_size_stride(permute_648, (128, 512), (512, 1))
    assert_size_stride(le_11, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_652, (512, 128), (128, 1))
    assert_size_stride(permute_656, (128, 512), (512, 1))
    assert_size_stride(le_12, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_660, (512, 128), (128, 1))
    assert_size_stride(permute_664, (128, 128), (128, 1))
    assert_size_stride(permute_677, (128, 512), (512, 1))
    assert_size_stride(permute_681, (128, 128), (128, 1))
    assert_size_stride(permute_685, (128, 128), (128, 1))
    assert_size_stride(permute_689, (128, 512), (512, 1))
    assert_size_stride(permute_693, (128, 512), (512, 1))
    assert_size_stride(permute_697, (512, 128), (128, 1))
    assert_size_stride(permute_701, (128, 512), (512, 1))
    assert_size_stride(le_13, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_705, (512, 128), (128, 1))
    assert_size_stride(permute_709, (128, 512), (512, 1))
    assert_size_stride(le_14, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_713, (512, 128), (128, 1))
    assert_size_stride(permute_717, (128, 512), (512, 1))
    assert_size_stride(le_15, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_721, (512, 128), (128, 1))
    assert_size_stride(permute_725, (128, 512), (512, 1))
    assert_size_stride(le_16, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_729, (512, 128), (128, 1))
    assert_size_stride(permute_733, (128, 128), (128, 1))
    assert_size_stride(permute_746, (128, 512), (512, 1))
    assert_size_stride(permute_750, (128, 128), (128, 1))
    assert_size_stride(permute_754, (128, 128), (128, 1))
    assert_size_stride(permute_758, (128, 512), (512, 1))
    assert_size_stride(permute_762, (128, 512), (512, 1))
    assert_size_stride(permute_766, (512, 128), (128, 1))
    assert_size_stride(permute_770, (128, 512), (512, 1))
    assert_size_stride(le_17, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_774, (512, 128), (128, 1))
    assert_size_stride(permute_778, (128, 512), (512, 1))
    assert_size_stride(le_18, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_782, (512, 128), (128, 1))
    assert_size_stride(permute_786, (128, 512), (512, 1))
    assert_size_stride(le_19, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_790, (512, 128), (128, 1))
    assert_size_stride(permute_794, (128, 512), (512, 1))
    assert_size_stride(le_20, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_798, (512, 128), (128, 1))
    assert_size_stride(permute_802, (128, 128), (128, 1))
    assert_size_stride(permute_815, (128, 512), (512, 1))
    assert_size_stride(permute_819, (128, 128), (128, 1))
    assert_size_stride(permute_823, (128, 128), (128, 1))
    assert_size_stride(permute_827, (128, 512), (512, 1))
    assert_size_stride(permute_831, (128, 512), (512, 1))
    assert_size_stride(permute_835, (512, 128), (128, 1))
    assert_size_stride(permute_839, (128, 512), (512, 1))
    assert_size_stride(le_21, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_843, (512, 128), (128, 1))
    assert_size_stride(permute_847, (128, 512), (512, 1))
    assert_size_stride(le_22, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_851, (512, 128), (128, 1))
    assert_size_stride(permute_855, (128, 512), (512, 1))
    assert_size_stride(le_23, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_859, (512, 128), (128, 1))
    assert_size_stride(permute_863, (128, 512), (512, 1))
    assert_size_stride(le_24, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_867, (512, 128), (128, 1))
    assert_size_stride(permute_871, (128, 128), (128, 1))
    assert_size_stride(permute_884, (128, 512), (512, 1))
    assert_size_stride(permute_888, (128, 128), (128, 1))
    assert_size_stride(permute_892, (128, 128), (128, 1))
    assert_size_stride(permute_896, (128, 512), (512, 1))
    assert_size_stride(permute_900, (128, 512), (512, 1))
    assert_size_stride(permute_904, (512, 128), (128, 1))
    assert_size_stride(permute_908, (128, 512), (512, 1))
    assert_size_stride(le_25, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_912, (512, 128), (128, 1))
    assert_size_stride(permute_916, (128, 512), (512, 1))
    assert_size_stride(le_26, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_920, (512, 128), (128, 1))
    assert_size_stride(permute_924, (128, 512), (512, 1))
    assert_size_stride(le_27, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_928, (512, 128), (128, 1))
    assert_size_stride(permute_932, (128, 512), (512, 1))
    assert_size_stride(le_28, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_936, (512, 128), (128, 1))
    assert_size_stride(permute_940, (128, 128), (128, 1))
    assert_size_stride(permute_953, (128, 512), (512, 1))
    assert_size_stride(permute_957, (128, 128), (128, 1))
    assert_size_stride(permute_961, (128, 128), (128, 1))
    assert_size_stride(permute_965, (128, 512), (512, 1))
    assert_size_stride(permute_969, (128, 512), (512, 1))
    assert_size_stride(permute_973, (512, 128), (128, 1))
    assert_size_stride(permute_977, (128, 512), (512, 1))
    assert_size_stride(le_29, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_981, (512, 128), (128, 1))
    assert_size_stride(permute_985, (128, 512), (512, 1))
    assert_size_stride(le_30, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_989, (512, 128), (128, 1))
    assert_size_stride(permute_993, (128, 512), (512, 1))
    assert_size_stride(le_31, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_997, (512, 128), (128, 1))
    assert_size_stride(permute_1001, (128, 512), (512, 1))
    assert_size_stride(le_32, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1005, (512, 128), (128, 1))
    assert_size_stride(permute_1009, (128, 128), (128, 1))
    assert_size_stride(permute_1022, (128, 512), (512, 1))
    assert_size_stride(permute_1026, (128, 128), (128, 1))
    assert_size_stride(permute_1030, (128, 128), (128, 1))
    assert_size_stride(permute_1034, (128, 512), (512, 1))
    assert_size_stride(permute_1038, (128, 512), (512, 1))
    assert_size_stride(permute_1042, (512, 128), (128, 1))
    assert_size_stride(permute_1046, (128, 512), (512, 1))
    assert_size_stride(le_33, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1050, (512, 128), (128, 1))
    assert_size_stride(permute_1054, (128, 512), (512, 1))
    assert_size_stride(le_34, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1058, (512, 128), (128, 1))
    assert_size_stride(permute_1062, (128, 512), (512, 1))
    assert_size_stride(le_35, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1066, (512, 128), (128, 1))
    assert_size_stride(permute_1070, (128, 512), (512, 1))
    assert_size_stride(le_36, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1074, (512, 128), (128, 1))
    assert_size_stride(permute_1078, (128, 128), (128, 1))
    assert_size_stride(permute_1091, (128, 512), (512, 1))
    assert_size_stride(permute_1095, (128, 128), (128, 1))
    assert_size_stride(permute_1099, (128, 128), (128, 1))
    assert_size_stride(permute_1103, (128, 512), (512, 1))
    assert_size_stride(permute_1107, (128, 512), (512, 1))
    assert_size_stride(permute_1111, (512, 128), (128, 1))
    assert_size_stride(permute_1115, (128, 512), (512, 1))
    assert_size_stride(le_37, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1119, (512, 128), (128, 1))
    assert_size_stride(permute_1123, (128, 512), (512, 1))
    assert_size_stride(le_38, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1127, (512, 128), (128, 1))
    assert_size_stride(permute_1131, (128, 512), (512, 1))
    assert_size_stride(le_39, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1135, (512, 128), (128, 1))
    assert_size_stride(permute_1139, (128, 512), (512, 1))
    assert_size_stride(le_40, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1143, (512, 128), (128, 1))
    assert_size_stride(permute_1147, (128, 128), (128, 1))
    assert_size_stride(permute_1160, (128, 512), (512, 1))
    assert_size_stride(permute_1164, (128, 128), (128, 1))
    assert_size_stride(permute_1168, (128, 128), (128, 1))
    assert_size_stride(permute_1172, (128, 512), (512, 1))
    assert_size_stride(permute_1176, (128, 512), (512, 1))
    assert_size_stride(permute_1180, (512, 128), (128, 1))
    assert_size_stride(permute_1184, (128, 512), (512, 1))
    assert_size_stride(le_41, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1188, (512, 128), (128, 1))
    assert_size_stride(permute_1192, (128, 512), (512, 1))
    assert_size_stride(le_42, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1196, (512, 128), (128, 1))
    assert_size_stride(permute_1200, (128, 512), (512, 1))
    assert_size_stride(le_43, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1204, (512, 128), (128, 1))
    assert_size_stride(permute_1208, (128, 512), (512, 1))
    assert_size_stride(le_44, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1212, (512, 128), (128, 1))
    assert_size_stride(permute_1216, (128, 128), (128, 1))
    assert_size_stride(permute_1229, (128, 512), (512, 1))
    assert_size_stride(permute_1233, (128, 128), (128, 1))
    assert_size_stride(permute_1237, (128, 128), (128, 1))
    assert_size_stride(permute_1241, (128, 512), (512, 1))
    assert_size_stride(permute_1245, (128, 512), (512, 1))
    assert_size_stride(permute_1249, (512, 128), (128, 1))
    assert_size_stride(permute_1253, (128, 512), (512, 1))
    assert_size_stride(le_45, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1257, (512, 128), (128, 1))
    assert_size_stride(permute_1261, (128, 512), (512, 1))
    assert_size_stride(le_46, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1265, (512, 128), (128, 1))
    assert_size_stride(permute_1269, (128, 512), (512, 1))
    assert_size_stride(le_47, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1273, (512, 128), (128, 1))
    assert_size_stride(permute_1277, (128, 512), (512, 1))
    assert_size_stride(le_48, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1281, (512, 128), (128, 1))
    assert_size_stride(permute_1285, (128, 128), (128, 1))
    assert_size_stride(permute_1298, (128, 512), (512, 1))
    assert_size_stride(permute_1302, (128, 128), (128, 1))
    assert_size_stride(permute_1306, (128, 128), (128, 1))
    assert_size_stride(permute_1310, (128, 512), (512, 1))
    assert_size_stride(permute_1314, (128, 512), (512, 1))
    assert_size_stride(permute_1318, (512, 128), (128, 1))
    assert_size_stride(permute_1322, (128, 512), (512, 1))
    assert_size_stride(le_49, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1326, (512, 128), (128, 1))
    assert_size_stride(permute_1330, (128, 512), (512, 1))
    assert_size_stride(le_50, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1334, (512, 128), (128, 1))
    assert_size_stride(permute_1338, (128, 512), (512, 1))
    assert_size_stride(le_51, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1342, (512, 128), (128, 1))
    assert_size_stride(permute_1346, (128, 512), (512, 1))
    assert_size_stride(le_52, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1350, (512, 128), (128, 1))
    assert_size_stride(permute_1354, (128, 128), (128, 1))
    assert_size_stride(permute_1367, (128, 512), (512, 1))
    assert_size_stride(permute_1371, (128, 128), (128, 1))
    assert_size_stride(permute_1375, (128, 128), (128, 1))
    assert_size_stride(permute_1379, (128, 512), (512, 1))
    assert_size_stride(permute_1383, (128, 512), (512, 1))
    assert_size_stride(permute_1387, (512, 128), (128, 1))
    assert_size_stride(permute_1391, (128, 512), (512, 1))
    assert_size_stride(le_53, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1395, (512, 128), (128, 1))
    assert_size_stride(permute_1399, (128, 512), (512, 1))
    assert_size_stride(le_54, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1403, (512, 128), (128, 1))
    assert_size_stride(permute_1407, (128, 512), (512, 1))
    assert_size_stride(le_55, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1411, (512, 128), (128, 1))
    assert_size_stride(permute_1415, (128, 512), (512, 1))
    assert_size_stride(le_56, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1419, (512, 128), (128, 1))
    assert_size_stride(permute_1423, (128, 128), (128, 1))
    assert_size_stride(permute_1436, (128, 512), (512, 1))
    assert_size_stride(permute_1440, (128, 128), (128, 1))
    assert_size_stride(permute_1444, (128, 128), (128, 1))
    assert_size_stride(permute_1448, (128, 512), (512, 1))
    assert_size_stride(permute_1452, (128, 512), (512, 1))
    assert_size_stride(permute_1456, (512, 128), (128, 1))
    assert_size_stride(permute_1460, (128, 512), (512, 1))
    assert_size_stride(le_57, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1464, (512, 128), (128, 1))
    assert_size_stride(permute_1468, (128, 512), (512, 1))
    assert_size_stride(le_58, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1472, (512, 128), (128, 1))
    assert_size_stride(permute_1476, (128, 512), (512, 1))
    assert_size_stride(le_59, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1480, (512, 128), (128, 1))
    assert_size_stride(permute_1484, (128, 512), (512, 1))
    assert_size_stride(le_60, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1488, (512, 128), (128, 1))
    assert_size_stride(permute_1492, (128, 128), (128, 1))
    assert_size_stride(permute_1505, (128, 512), (512, 1))
    assert_size_stride(permute_1509, (128, 128), (128, 1))
    assert_size_stride(permute_1513, (128, 128), (128, 1))
    assert_size_stride(permute_1517, (128, 512), (512, 1))
    assert_size_stride(permute_1521, (128, 512), (512, 1))
    assert_size_stride(permute_1525, (512, 128), (128, 1))
    assert_size_stride(permute_1529, (128, 512), (512, 1))
    assert_size_stride(le_61, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1533, (512, 128), (128, 1))
    assert_size_stride(permute_1537, (128, 512), (512, 1))
    assert_size_stride(le_62, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1541, (512, 128), (128, 1))
    assert_size_stride(permute_1545, (128, 512), (512, 1))
    assert_size_stride(le_63, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1549, (512, 128), (128, 1))
    assert_size_stride(permute_1553, (128, 512), (512, 1))
    assert_size_stride(le_64, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1557, (512, 128), (128, 1))
    assert_size_stride(permute_1561, (128, 128), (128, 1))
    assert_size_stride(permute_1574, (128, 512), (512, 1))
    assert_size_stride(permute_1578, (128, 128), (128, 1))
    assert_size_stride(permute_1582, (128, 128), (128, 1))
    assert_size_stride(permute_1586, (128, 512), (512, 1))
    assert_size_stride(permute_1590, (128, 512), (512, 1))
    assert_size_stride(permute_1594, (512, 128), (128, 1))
    assert_size_stride(permute_1598, (128, 512), (512, 1))
    assert_size_stride(le_65, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1602, (512, 128), (128, 1))
    assert_size_stride(permute_1606, (128, 512), (512, 1))
    assert_size_stride(le_66, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1610, (512, 128), (128, 1))
    assert_size_stride(permute_1614, (128, 512), (512, 1))
    assert_size_stride(le_67, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1618, (512, 128), (128, 1))
    assert_size_stride(permute_1622, (128, 512), (512, 1))
    assert_size_stride(le_68, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1626, (512, 128), (128, 1))
    assert_size_stride(permute_1630, (128, 128), (128, 1))
    assert_size_stride(permute_1643, (128, 512), (512, 1))
    assert_size_stride(permute_1647, (128, 128), (128, 1))
    assert_size_stride(permute_1651, (128, 128), (128, 1))
    assert_size_stride(permute_1655, (128, 512), (512, 1))
    assert_size_stride(permute_1659, (128, 512), (512, 1))
    assert_size_stride(permute_1663, (512, 128), (128, 1))
    assert_size_stride(permute_1667, (128, 512), (512, 1))
    assert_size_stride(le_69, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1671, (512, 128), (128, 1))
    assert_size_stride(permute_1675, (128, 512), (512, 1))
    assert_size_stride(le_70, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1679, (512, 128), (128, 1))
    assert_size_stride(permute_1683, (128, 512), (512, 1))
    assert_size_stride(le_71, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1687, (512, 128), (128, 1))
    assert_size_stride(permute_1691, (128, 512), (512, 1))
    assert_size_stride(le_72, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1695, (512, 128), (128, 1))
    assert_size_stride(permute_1699, (128, 128), (128, 1))
    assert_size_stride(permute_1712, (128, 512), (512, 1))
    assert_size_stride(permute_1716, (128, 128), (128, 1))
    assert_size_stride(permute_1720, (128, 128), (128, 1))
    assert_size_stride(permute_1724, (128, 512), (512, 1))
    assert_size_stride(permute_1728, (128, 512), (512, 1))
    assert_size_stride(permute_1732, (512, 128), (128, 1))
    assert_size_stride(permute_1736, (128, 512), (512, 1))
    assert_size_stride(le_73, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1740, (512, 128), (128, 1))
    assert_size_stride(permute_1744, (128, 512), (512, 1))
    assert_size_stride(le_74, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1748, (512, 128), (128, 1))
    assert_size_stride(permute_1752, (128, 512), (512, 1))
    assert_size_stride(le_75, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1756, (512, 128), (128, 1))
    assert_size_stride(permute_1760, (128, 512), (512, 1))
    assert_size_stride(le_76, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1764, (512, 128), (128, 1))
    assert_size_stride(permute_1768, (128, 128), (128, 1))
    assert_size_stride(permute_1781, (128, 512), (512, 1))
    assert_size_stride(permute_1785, (128, 128), (128, 1))
    assert_size_stride(permute_1789, (128, 128), (128, 1))
    assert_size_stride(permute_1793, (128, 512), (512, 1))
    assert_size_stride(permute_1797, (128, 512), (512, 1))
    assert_size_stride(permute_1801, (512, 128), (128, 1))
    assert_size_stride(permute_1805, (128, 512), (512, 1))
    assert_size_stride(le_77, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1809, (512, 128), (128, 1))
    assert_size_stride(permute_1813, (128, 512), (512, 1))
    assert_size_stride(le_78, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1817, (512, 128), (128, 1))
    assert_size_stride(permute_1821, (128, 512), (512, 1))
    assert_size_stride(le_79, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1825, (512, 128), (128, 1))
    assert_size_stride(permute_1829, (128, 512), (512, 1))
    assert_size_stride(le_80, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1833, (512, 128), (128, 1))
    assert_size_stride(permute_1837, (128, 128), (128, 1))
    assert_size_stride(permute_1850, (128, 512), (512, 1))
    assert_size_stride(permute_1854, (128, 128), (128, 1))
    assert_size_stride(permute_1858, (128, 128), (128, 1))
    assert_size_stride(permute_1862, (128, 512), (512, 1))
    assert_size_stride(permute_1866, (128, 512), (512, 1))
    assert_size_stride(permute_1870, (512, 128), (128, 1))
    assert_size_stride(permute_1874, (128, 512), (512, 1))
    assert_size_stride(le_81, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1878, (512, 128), (128, 1))
    assert_size_stride(permute_1882, (128, 512), (512, 1))
    assert_size_stride(le_82, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1886, (512, 128), (128, 1))
    assert_size_stride(permute_1890, (128, 512), (512, 1))
    assert_size_stride(le_83, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1894, (512, 128), (128, 1))
    assert_size_stride(permute_1898, (128, 512), (512, 1))
    assert_size_stride(le_84, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1902, (512, 128), (128, 1))
    assert_size_stride(permute_1906, (128, 128), (128, 1))
    assert_size_stride(permute_1919, (128, 512), (512, 1))
    assert_size_stride(permute_1923, (128, 128), (128, 1))
    assert_size_stride(permute_1927, (128, 128), (128, 1))
    assert_size_stride(permute_1931, (128, 512), (512, 1))
    assert_size_stride(permute_1935, (128, 512), (512, 1))
    assert_size_stride(permute_1939, (512, 128), (128, 1))
    assert_size_stride(permute_1943, (128, 512), (512, 1))
    assert_size_stride(le_85, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1947, (512, 128), (128, 1))
    assert_size_stride(permute_1951, (128, 512), (512, 1))
    assert_size_stride(le_86, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1955, (512, 128), (128, 1))
    assert_size_stride(permute_1959, (128, 512), (512, 1))
    assert_size_stride(le_87, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1963, (512, 128), (128, 1))
    assert_size_stride(permute_1967, (128, 512), (512, 1))
    assert_size_stride(le_88, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1971, (512, 128), (128, 1))
    assert_size_stride(permute_1975, (128, 128), (128, 1))
    assert_size_stride(permute_1988, (128, 512), (512, 1))
    assert_size_stride(permute_1992, (128, 128), (128, 1))
    assert_size_stride(permute_1996, (128, 128), (128, 1))
    assert_size_stride(permute_2000, (128, 512), (512, 1))
    assert_size_stride(permute_2004, (128, 512), (512, 1))
    assert_size_stride(permute_2008, (512, 128), (128, 1))
    assert_size_stride(permute_2012, (128, 512), (512, 1))
    assert_size_stride(le_89, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_2016, (512, 128), (128, 1))
    assert_size_stride(permute_2020, (128, 512), (512, 1))
    assert_size_stride(le_90, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_2024, (512, 128), (128, 1))
    assert_size_stride(permute_2028, (128, 512), (512, 1))
    assert_size_stride(le_91, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_2032, (512, 128), (128, 1))
    assert_size_stride(permute_2036, (128, 512), (512, 1))
    assert_size_stride(le_92, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_2040, (512, 128), (128, 1))
    assert_size_stride(permute_2044, (128, 128), (128, 1))
    assert_size_stride(permute_2057, (128, 512), (512, 1))
    assert_size_stride(permute_2061, (128, 128), (128, 1))
    assert_size_stride(permute_2065, (128, 128), (128, 1))
    assert_size_stride(permute_2069, (128, 512), (512, 1))
    assert_size_stride(permute_2073, (128, 512), (512, 1))
    assert_size_stride(permute_2077, (512, 128), (128, 1))
    assert_size_stride(permute_2081, (128, 512), (512, 1))
    assert_size_stride(le_93, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_2085, (512, 128), (128, 1))
    assert_size_stride(permute_2089, (128, 512), (512, 1))
    assert_size_stride(le_94, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_2093, (512, 128), (128, 1))
    assert_size_stride(permute_2097, (128, 512), (512, 1))
    assert_size_stride(le_95, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_2101, (512, 128), (128, 1))
    assert_size_stride(permute_2105, (128, 512), (512, 1))
    assert_size_stride(le_96, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_2109, (512, 128), (128, 1))
    assert_size_stride(permute_2113, (128, 128), (128, 1))
    assert_size_stride(permute_2126, (128, 512), (512, 1))
    assert_size_stride(permute_2130, (128, 128), (128, 1))
    assert_size_stride(permute_2134, (128, 128), (128, 1))
    assert_size_stride(permute_2138, (128, 512), (512, 1))
    assert_size_stride(permute_2142, (128, 512), (512, 1))
    assert_size_stride(permute_2146, (512, 384), (384, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 128, 30522), (3906816, 30522, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf1 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_10, add_12, add_6, add_8, attention_output, attention_output_1, attention_output_2, layer_input_4, mul_2, mul_4, mul_5, mul_6], Original ATen: [aten.add, aten.mul]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_add_mul_0.run(addmm_8, addmm_6, addmm_1, primals_3, primals_4, primals_7, primals_8, addmm_12, addmm_10, primals_9, primals_10, primals_11, primals_12, buf0, buf1, 16384, grid=grid(16384), stream=stream0)
        del addmm_12
        del addmm_8
        del primals_12
        del primals_8
        buf2 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf3 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_21, add_23, add_25, add_27, attention_output_5, attention_output_6, attention_output_7, layer_input_9, mul_10, mul_12, mul_13, mul_14], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_23, addmm_21, addmm_16, primals_19, primals_20, primals_23, primals_24, addmm_27, addmm_25, primals_25, primals_26, primals_27, primals_28, buf2, buf3, 16384, grid=grid(16384), stream=stream0)
        del addmm_23
        del addmm_27
        del primals_24
        del primals_28
        buf4 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf5 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_36, add_38, add_40, add_42, attention_output_10, attention_output_11, attention_output_12, layer_input_14, mul_18, mul_20, mul_21, mul_22], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_38, addmm_36, addmm_31, primals_35, primals_36, primals_39, primals_40, addmm_42, addmm_40, primals_41, primals_42, primals_43, primals_44, buf4, buf5, 16384, grid=grid(16384), stream=stream0)
        del addmm_38
        del addmm_42
        del primals_40
        del primals_44
        buf6 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        buf11 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        buf16 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        buf21 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        buf26 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        buf31 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        buf36 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        buf41 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        buf46 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        buf51 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        buf56 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_106, add_121, add_136, add_151, add_166, add_181, add_196, add_211, add_226, add_241, add_256, add_271, add_286, add_301, add_31, add_316, add_331, add_346, add_46, add_61, add_76, add_91, mul_105, mul_113, mul_121, mul_129, mul_137, mul_145, mul_153, mul_161, mul_169, mul_17, mul_177, mul_25, mul_33, mul_41, mul_49, mul_57, mul_65, mul_73, mul_81, mul_89, mul_9, mul_97, value_tensor_1, value_tensor_10, value_tensor_11, value_tensor_12, value_tensor_13, value_tensor_14, value_tensor_15, value_tensor_16, value_tensor_17, value_tensor_18, value_tensor_19, value_tensor_2, value_tensor_20, value_tensor_21, value_tensor_22, value_tensor_3, value_tensor_4, value_tensor_5, value_tensor_6, value_tensor_7, value_tensor_8, value_tensor_9], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_1.run(addmm_45, addmm_30, add_16, primals_17, primals_18, primals_33, primals_34, addmm_75, addmm_60, primals_49, primals_50, primals_65, primals_66, addmm_105, addmm_90, primals_81, primals_82, primals_97, primals_98, addmm_135, addmm_120, primals_113, primals_114, primals_129, primals_130, addmm_165, addmm_150, primals_145, primals_146, primals_161, primals_162, addmm_195, addmm_180, primals_177, primals_178, primals_193, primals_194, addmm_225, addmm_210, primals_209, primals_210, primals_225, primals_226, addmm_255, addmm_240, primals_241, primals_242, primals_257, primals_258, addmm_285, addmm_270, primals_273, primals_274, primals_289, primals_290, addmm_315, addmm_300, primals_305, primals_306, primals_321, primals_322, addmm_345, addmm_330, primals_337, primals_338, primals_353, primals_354, buf6, buf11, buf16, buf21, buf26, buf31, buf36, buf41, buf46, buf51, buf56, 65536, grid=grid(65536), stream=stream0)
        del addmm_105
        del addmm_135
        del addmm_165
        del addmm_195
        del addmm_225
        del addmm_255
        del addmm_285
        del addmm_315
        del addmm_345
        del addmm_45
        del addmm_75
        del primals_130
        del primals_162
        del primals_194
        del primals_226
        del primals_258
        del primals_290
        del primals_322
        del primals_34
        del primals_354
        del primals_66
        del primals_98
        buf7 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf8 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_51, add_53, add_55, add_57, attention_output_15, attention_output_16, attention_output_17, layer_input_19, mul_26, mul_28, mul_29, mul_30], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_53, addmm_51, addmm_46, primals_51, primals_52, primals_55, primals_56, addmm_57, addmm_55, primals_57, primals_58, primals_59, primals_60, buf7, buf8, 16384, grid=grid(16384), stream=stream0)
        del addmm_53
        del addmm_57
        del primals_56
        del primals_60
        buf9 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf10 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_66, add_68, add_70, add_72, attention_output_20, attention_output_21, attention_output_22, layer_input_24, mul_34, mul_36, mul_37, mul_38], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_68, addmm_66, addmm_61, primals_67, primals_68, primals_71, primals_72, addmm_72, addmm_70, primals_73, primals_74, primals_75, primals_76, buf9, buf10, 16384, grid=grid(16384), stream=stream0)
        del addmm_68
        del addmm_72
        del primals_72
        del primals_76
        buf12 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf13 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_81, add_83, add_85, add_87, attention_output_25, attention_output_26, attention_output_27, layer_input_29, mul_42, mul_44, mul_45, mul_46], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_83, addmm_81, addmm_76, primals_83, primals_84, primals_87, primals_88, addmm_87, addmm_85, primals_89, primals_90, primals_91, primals_92, buf12, buf13, 16384, grid=grid(16384), stream=stream0)
        del addmm_83
        del addmm_87
        del primals_88
        del primals_92
        buf14 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf15 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_100, add_102, add_96, add_98, attention_output_30, attention_output_31, attention_output_32, layer_input_34, mul_50, mul_52, mul_53, mul_54], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_98, addmm_96, addmm_91, primals_99, primals_100, primals_103, primals_104, addmm_102, addmm_100, primals_105, primals_106, primals_107, primals_108, buf14, buf15, 16384, grid=grid(16384), stream=stream0)
        del addmm_102
        del addmm_98
        del primals_104
        del primals_108
        buf17 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf18 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_111, add_113, add_115, add_117, attention_output_35, attention_output_36, attention_output_37, layer_input_39, mul_58, mul_60, mul_61, mul_62], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_113, addmm_111, addmm_106, primals_115, primals_116, primals_119, primals_120, addmm_117, addmm_115, primals_121, primals_122, primals_123, primals_124, buf17, buf18, 16384, grid=grid(16384), stream=stream0)
        del addmm_113
        del addmm_117
        del primals_120
        del primals_124
        buf19 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf20 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_126, add_128, add_130, add_132, attention_output_40, attention_output_41, attention_output_42, layer_input_44, mul_66, mul_68, mul_69, mul_70], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_128, addmm_126, addmm_121, primals_131, primals_132, primals_135, primals_136, addmm_132, addmm_130, primals_137, primals_138, primals_139, primals_140, buf19, buf20, 16384, grid=grid(16384), stream=stream0)
        del addmm_128
        del addmm_132
        del primals_136
        del primals_140
        buf22 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf23 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_141, add_143, add_145, add_147, attention_output_45, attention_output_46, attention_output_47, layer_input_49, mul_74, mul_76, mul_77, mul_78], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_143, addmm_141, addmm_136, primals_147, primals_148, primals_151, primals_152, addmm_147, addmm_145, primals_153, primals_154, primals_155, primals_156, buf22, buf23, 16384, grid=grid(16384), stream=stream0)
        del addmm_143
        del addmm_147
        del primals_152
        del primals_156
        buf24 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf25 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_156, add_158, add_160, add_162, attention_output_50, attention_output_51, attention_output_52, layer_input_54, mul_82, mul_84, mul_85, mul_86], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_158, addmm_156, addmm_151, primals_163, primals_164, primals_167, primals_168, addmm_162, addmm_160, primals_169, primals_170, primals_171, primals_172, buf24, buf25, 16384, grid=grid(16384), stream=stream0)
        del addmm_158
        del addmm_162
        del primals_168
        del primals_172
        buf27 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf28 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_171, add_173, add_175, add_177, attention_output_55, attention_output_56, attention_output_57, layer_input_59, mul_90, mul_92, mul_93, mul_94], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_173, addmm_171, addmm_166, primals_179, primals_180, primals_183, primals_184, addmm_177, addmm_175, primals_185, primals_186, primals_187, primals_188, buf27, buf28, 16384, grid=grid(16384), stream=stream0)
        del addmm_173
        del addmm_177
        del primals_184
        del primals_188
        buf29 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf30 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_186, add_188, add_190, add_192, attention_output_60, attention_output_61, attention_output_62, layer_input_64, mul_100, mul_101, mul_102, mul_98], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_188, addmm_186, addmm_181, primals_195, primals_196, primals_199, primals_200, addmm_192, addmm_190, primals_201, primals_202, primals_203, primals_204, buf29, buf30, 16384, grid=grid(16384), stream=stream0)
        del addmm_188
        del addmm_192
        del primals_200
        del primals_204
        buf32 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf33 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_201, add_203, add_205, add_207, attention_output_65, attention_output_66, attention_output_67, layer_input_69, mul_106, mul_108, mul_109, mul_110], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_203, addmm_201, addmm_196, primals_211, primals_212, primals_215, primals_216, addmm_207, addmm_205, primals_217, primals_218, primals_219, primals_220, buf32, buf33, 16384, grid=grid(16384), stream=stream0)
        del addmm_203
        del addmm_207
        del primals_216
        del primals_220
        buf34 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf35 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_216, add_218, add_220, add_222, attention_output_70, attention_output_71, attention_output_72, layer_input_74, mul_114, mul_116, mul_117, mul_118], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_218, addmm_216, addmm_211, primals_227, primals_228, primals_231, primals_232, addmm_222, addmm_220, primals_233, primals_234, primals_235, primals_236, buf34, buf35, 16384, grid=grid(16384), stream=stream0)
        del addmm_218
        del addmm_222
        del primals_232
        del primals_236
        buf37 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf38 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_231, add_233, add_235, add_237, attention_output_75, attention_output_76, attention_output_77, layer_input_79, mul_122, mul_124, mul_125, mul_126], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_233, addmm_231, addmm_226, primals_243, primals_244, primals_247, primals_248, addmm_237, addmm_235, primals_249, primals_250, primals_251, primals_252, buf37, buf38, 16384, grid=grid(16384), stream=stream0)
        del addmm_233
        del addmm_237
        del primals_248
        del primals_252
        buf39 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf40 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_246, add_248, add_250, add_252, attention_output_80, attention_output_81, attention_output_82, layer_input_84, mul_130, mul_132, mul_133, mul_134], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_248, addmm_246, addmm_241, primals_259, primals_260, primals_263, primals_264, addmm_252, addmm_250, primals_265, primals_266, primals_267, primals_268, buf39, buf40, 16384, grid=grid(16384), stream=stream0)
        del addmm_248
        del addmm_252
        del primals_264
        del primals_268
        buf42 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf43 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_261, add_263, add_265, add_267, attention_output_85, attention_output_86, attention_output_87, layer_input_89, mul_138, mul_140, mul_141, mul_142], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_263, addmm_261, addmm_256, primals_275, primals_276, primals_279, primals_280, addmm_267, addmm_265, primals_281, primals_282, primals_283, primals_284, buf42, buf43, 16384, grid=grid(16384), stream=stream0)
        del addmm_263
        del addmm_267
        del primals_280
        del primals_284
        buf44 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf45 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_276, add_278, add_280, add_282, attention_output_90, attention_output_91, attention_output_92, layer_input_94, mul_146, mul_148, mul_149, mul_150], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_278, addmm_276, addmm_271, primals_291, primals_292, primals_295, primals_296, addmm_282, addmm_280, primals_297, primals_298, primals_299, primals_300, buf44, buf45, 16384, grid=grid(16384), stream=stream0)
        del addmm_278
        del addmm_282
        del primals_296
        del primals_300
        buf47 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf48 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_291, add_293, add_295, add_297, attention_output_95, attention_output_96, attention_output_97, layer_input_99, mul_154, mul_156, mul_157, mul_158], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_293, addmm_291, addmm_286, primals_307, primals_308, primals_311, primals_312, addmm_297, addmm_295, primals_313, primals_314, primals_315, primals_316, buf47, buf48, 16384, grid=grid(16384), stream=stream0)
        del addmm_293
        del addmm_297
        del primals_312
        del primals_316
        buf49 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf50 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_306, add_308, add_310, add_312, attention_output_100, attention_output_101, attention_output_102, layer_input_104, mul_162, mul_164, mul_165, mul_166], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_308, addmm_306, addmm_301, primals_323, primals_324, primals_327, primals_328, addmm_312, addmm_310, primals_329, primals_330, primals_331, primals_332, buf49, buf50, 16384, grid=grid(16384), stream=stream0)
        del addmm_308
        del addmm_312
        del primals_328
        del primals_332
        buf52 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf53 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_321, add_323, add_325, add_327, attention_output_105, attention_output_106, attention_output_107, layer_input_109, mul_170, mul_172, mul_173, mul_174], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_323, addmm_321, addmm_316, primals_339, primals_340, primals_343, primals_344, addmm_327, addmm_325, primals_345, primals_346, primals_347, primals_348, buf52, buf53, 16384, grid=grid(16384), stream=stream0)
        del addmm_323
        del addmm_327
        del primals_344
        del primals_348
        buf54 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf55 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_336, add_338, add_340, add_342, attention_output_110, attention_output_111, attention_output_112, layer_input_114, mul_178, mul_180, mul_181, mul_182], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_0.run(addmm_338, addmm_336, addmm_331, primals_355, primals_356, primals_359, primals_360, addmm_342, addmm_340, primals_361, primals_362, primals_363, primals_364, buf54, buf55, 16384, grid=grid(16384), stream=stream0)
        del addmm_338
        del addmm_342
        del primals_360
        del primals_364
        buf59 = empty((128, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_2.run(buf59, 3906816, grid=grid(3906816), stream=stream0)
        buf60 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.int64)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_3.run(primals_1121, buf60, 128, grid=grid(128), stream=stream0)
        aten.scatter_(buf59,1,buf60,-1.0)
        del buf60
        buf63 = empty_strided((128, 1, 4), (4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_4.run(buf59, primals_1121, tangents_1, convert_element_type, buf63, 512, 7631, grid=grid(512), stream=stream0)
        buf64 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_per_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_5.run(buf63, buf64, 128, 4, grid=grid(128), stream=stream0)
        buf67 = empty((3906816, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_2.run(buf67, 3906816, grid=grid(3906816), stream=stream0)
        buf62 = empty((128, 30522), device='cuda', dtype=torch.float32)
        buf65 = buf62; del buf62  # reuse
        buf70 = empty((128, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.squeeze]
        triton_poi_fused_add_as_strided_scatter_squeeze_6.run(buf65, tangents_2, buf59, primals_1121, tangents_1, convert_element_type, sub_27, buf64, buf67, buf70, 3906816, grid=grid(3906816), stream=stream0)
        del buf59
        del buf65
        del convert_element_type
        del primals_1121
        del sub_27
        del tangents_1
        del tangents_2
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_7.run(buf67, buf70, 3906816, grid=grid(3906816), stream=stream0)
        buf74 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf70, permute_484, out=buf74)
        del permute_484
        buf79 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_217, hidden_states_219, loss], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.nll_loss_forward, aten.relu, aten.threshold_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_nll_loss_forward_relu_threshold_backward_8.run(buf74, primals_1117, addmm_361, getitem_49, rsqrt, buf79, 128, 512, grid=grid(128), stream=stream0)
        del primals_1117
        buf80 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (128, 512), (512, 1), 0), permute_486, out=buf80)
        del permute_486
        buf85 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_9.run(buf80, primals_385, buf85, 65536, grid=grid(65536), stream=stream0)
        del primals_385
        buf86 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (128, 512), (512, 1), 0), permute_490, out=buf86)
        del permute_490
        buf91 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf86, primals_383, buf91, 16384, grid=grid(16384), stream=stream0)
        del primals_383
        buf92 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (128, 128), (128, 1), 0), permute_494, out=buf92)
        del permute_494
        buf95 = reinterpret_tensor(buf92, (1, 128, 512), (65536, 512, 1), 0); del buf92  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf95, le_1, 65536, grid=grid(65536), stream=stream0)
        del le_1
        buf96 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf95, (128, 512), (512, 1), 0), permute_498, out=buf96)
        del permute_498
        buf101 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf91, buf96, primals_381, buf101, 16384, grid=grid(16384), stream=stream0)
        buf102 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (128, 128), (128, 1), 0), permute_502, out=buf102)
        del permute_502
        buf105 = reinterpret_tensor(buf102, (1, 128, 512), (65536, 512, 1), 0); del buf102  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf105, le_2, 65536, grid=grid(65536), stream=stream0)
        del le_2
        buf106 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (128, 512), (512, 1), 0), permute_506, out=buf106)
        del permute_506
        buf57 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf58 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf111 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_351, add_353, add_355, add_357, attention_output_115, attention_output_116, attention_output_117, layer_input_119, mul_186, mul_188, mul_189, mul_190], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_13.run(addmm_353, addmm_351, addmm_346, primals_371, primals_372, primals_375, primals_376, addmm_357, addmm_355, primals_377, primals_378, primals_379, primals_380, buf101, buf106, buf57, buf58, buf111, 16384, grid=grid(16384), stream=stream0)
        del addmm_353
        del addmm_357
        del primals_376
        del primals_379
        del primals_380
        buf72 = empty((1, 1, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf67, buf72, 30522, 128, grid=grid(30522), stream=stream0)
        del buf67
        buf73 = empty((512, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_483, buf70, out=buf73)
        del permute_483
        buf77 = reinterpret_tensor(buf63, (512, ), (1, ), 0); del buf63  # reuse
        buf78 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_217, hidden_states_219], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.relu]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_relu_15.run(buf74, addmm_361, getitem_49, rsqrt, buf77, buf78, 512, 128, grid=grid(512), stream=stream0)
        del addmm_361
        del getitem_49
        del rsqrt
        buf81 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (512, 128), (1, 512), 0), view_962, out=buf81)
        del view_962
        buf82 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf79, buf82, 512, 128, grid=grid(512), stream=stream0)
        buf83 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf84 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_361, mul_185, value_tensor_23], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_17.run(buf80, addmm_360, buf56, primals_369, primals_370, buf83, buf84, 512, 128, grid=grid(512), stream=stream0)
        del addmm_360
        del primals_370
        buf87 = reinterpret_tensor(buf80, (512, 128), (128, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (512, 128), (1, 512), 0), view_960, out=buf87)
        del view_960
        buf88 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf85, buf88, 512, 128, grid=grid(512), stream=stream0)
        buf89 = reinterpret_tensor(buf64, (1, 1, 128), (128, 128, 1), 0); del buf64  # reuse
        buf90 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf94 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf99 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf100 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_359, attention_output_118, mul_191], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf86, addmm_359, buf58, primals_381, primals_382, buf91, buf96, buf89, buf90, buf94, buf99, buf100, 128, 128, grid=grid(128), stream=stream0)
        del addmm_359
        del buf58
        del buf86
        del buf96
        del primals_381
        del primals_382
        buf93 = reinterpret_tensor(buf79, (128, 512), (512, 1), 0); del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (128, 128), (1, 128), 0), view_958, out=buf93)
        del view_958
        buf97 = reinterpret_tensor(buf74, (512, 128), (128, 1), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf95, (512, 128), (1, 512), 0), view_956, out=buf97)
        del view_956
        buf98 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf95, buf98, 512, 128, grid=grid(512), stream=stream0)
        buf103 = reinterpret_tensor(buf95, (128, 512), (512, 1), 0); del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (128, 128), (1, 128), 0), view_954, out=buf103)
        del view_954
        buf112 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (128, 128), (128, 1), 0), permute_510, out=buf112)
        del permute_510
        buf115 = reinterpret_tensor(buf112, (1, 128, 512), (65536, 512, 1), 0); del buf112  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf115, le_3, 65536, grid=grid(65536), stream=stream0)
        del le_3
        buf116 = reinterpret_tensor(buf91, (128, 128), (128, 1), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (128, 512), (512, 1), 0), permute_514, out=buf116)
        del permute_514
        buf104 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf109 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf110 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf114 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf119 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf120 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_355, attention_output_116, mul_189], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf101, buf106, addmm_355, buf57, primals_377, primals_378, buf111, buf116, buf104, buf109, buf110, buf114, buf119, buf120, 128, 128, grid=grid(128), stream=stream0)
        del addmm_355
        del buf101
        del primals_378
        buf107 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (512, 128), (1, 512), 0), view_952, out=buf107)
        del view_952
        buf108 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf105, buf108, 512, 128, grid=grid(512), stream=stream0)
        buf113 = reinterpret_tensor(buf105, (128, 512), (512, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (128, 128), (1, 128), 0), view_950, out=buf113)
        del view_950
        buf117 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (512, 128), (1, 512), 0), view_948, out=buf117)
        del view_948
        buf118 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf115, buf118, 512, 128, grid=grid(512), stream=stream0)
        buf121 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf121, buf116, primals_377, 16384, grid=grid(16384), stream=stream0)
        del primals_377
        buf122 = reinterpret_tensor(buf115, (128, 512), (512, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (128, 128), (128, 1), 0), permute_518, out=buf122)
        del permute_518
        buf123 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (128, 128), (1, 128), 0), view_946, out=buf123)
        del view_946
        buf125 = reinterpret_tensor(buf122, (1, 128, 512), (65536, 512, 1), 0); del buf122  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf125, le_4, 65536, grid=grid(65536), stream=stream0)
        del le_4
        buf126 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (128, 512), (512, 1), 0), permute_522, out=buf126)
        del permute_522
        buf131 = buf57; del buf57  # reuse
        buf156 = reinterpret_tensor(buf106, (1, 128, 128), (16384, 128, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf121, buf126, primals_375, primals_371, buf131, buf156, 16384, grid=grid(16384), stream=stream0)
        del primals_375
        buf124 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf129 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf130 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf134 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf154 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf155 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_351, layer_input_119, mul_186], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf121, buf126, addmm_351, addmm_346, primals_371, primals_372, buf131, buf124, buf129, buf130, buf134, buf154, buf155, 128, 128, grid=grid(128), stream=stream0)
        del addmm_346
        del addmm_351
        del primals_371
        del primals_372
        buf127 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (512, 128), (1, 512), 0), view_944, out=buf127)
        del view_944
        buf128 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf125, buf128, 512, 128, grid=grid(512), stream=stream0)
        buf132 = buf126; del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf131, (128, 128), (128, 1), 0), permute_526, out=buf132)
        del permute_526
        buf133 = reinterpret_tensor(buf121, (128, 128), (128, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf131, (128, 128), (1, 128), 0), view_942, out=buf133)
        del view_942
        # Source Nodes: [], Original ATen: []
        buf135 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf132, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default, clone_default_1, clone_default_2, None, alias_default_1, getitem_51, getitem_52, getitem_53, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_1
        del clone_default
        del clone_default_1
        del clone_default_2
        del getitem_51
        del getitem_52
        del getitem_53
        buf136 = buf135[0]
        buf137 = buf135[1]
        buf138 = buf135[2]
        del buf135
        buf139 = reinterpret_tensor(buf125, (128, 512), (512, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf138, (128, 128), (128, 1), 0), permute_539, out=buf139)
        del permute_539
        buf140 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf138, (128, 128), (1, 128), 0), view_922, out=buf140)
        buf141 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf138, buf141, 128, 128, grid=grid(128), stream=stream0)
        buf142 = reinterpret_tensor(buf138, (128, 128), (128, 1), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf137, (128, 128), (128, 1), 0), permute_543, out=buf142)
        del permute_543
        buf143 = buf132; del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf137, (128, 128), (1, 128), 0), view_926, out=buf143)
        buf144 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf137, buf144, 128, 128, grid=grid(128), stream=stream0)
        buf145 = reinterpret_tensor(buf137, (128, 128), (128, 1), 0); del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf136, (128, 128), (128, 1), 0), permute_547, out=buf145)
        del permute_547
        buf146 = reinterpret_tensor(buf131, (128, 128), (128, 1), 0); del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf136, (128, 128), (1, 128), 0), view_926, out=buf146)
        del view_926
        buf147 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf136, buf147, 128, 128, grid=grid(128), stream=stream0)
        del buf136
        buf148 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf149 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf142, buf145, addmm_347, buf148, buf149, 128, 128, grid=grid(128), stream=stream0)
        del addmm_347
        buf150 = reinterpret_tensor(buf142, (1, 128, 128), (16384, 128, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf150, buf145, primals_373, 16384, grid=grid(16384), stream=stream0)
        del primals_373
        buf151 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf150, (128, 128), (128, 1), 0), permute_551, out=buf151)
        del permute_551
        buf152 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf150, (128, 128), (1, 128), 0), view_922, out=buf152)
        buf153 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf150, buf153, 128, 128, grid=grid(128), stream=stream0)
        buf157 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf156, (128, 128), (128, 1), 0), permute_555, out=buf157)
        del permute_555
        buf158 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf156, (128, 128), (1, 128), 0), view_922, out=buf158)
        del view_922
        buf159 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf156, buf159, 128, 128, grid=grid(128), stream=stream0)
        buf160 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf162 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_26.run(buf85, buf139, buf151, buf157, buf56, buf160, buf162, 512, 128, grid=grid(512), stream=stream0)
        buf161 = reinterpret_tensor(buf139, (1, 128, 512), (65536, 512, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_27.run(buf161, buf85, buf151, buf157, primals_369, 65536, grid=grid(65536), stream=stream0)
        del primals_369
        buf163 = reinterpret_tensor(buf156, (128, 128), (128, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (128, 512), (512, 1), 0), permute_559, out=buf163)
        del permute_559
        buf164 = reinterpret_tensor(buf85, (512, 128), (128, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (512, 128), (1, 512), 0), view_920, out=buf164)
        del view_920
        buf165 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf161, buf165, 512, 128, grid=grid(512), stream=stream0)
        buf168 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf163, primals_367, buf168, 16384, grid=grid(16384), stream=stream0)
        del primals_367
        buf169 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf168, (128, 128), (128, 1), 0), permute_563, out=buf169)
        del permute_563
        buf172 = reinterpret_tensor(buf169, (1, 128, 512), (65536, 512, 1), 0); del buf169  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf172, le_5, 65536, grid=grid(65536), stream=stream0)
        del le_5
        buf173 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf172, (128, 512), (512, 1), 0), permute_567, out=buf173)
        del permute_567
        buf166 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf167 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf171 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf176 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf177 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_344, attention_output_113, mul_183], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf163, addmm_344, buf55, primals_365, primals_366, buf168, buf173, buf166, buf167, buf171, buf176, buf177, 128, 128, grid=grid(128), stream=stream0)
        del addmm_344
        del primals_366
        buf170 = buf151; del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf168, (128, 128), (1, 128), 0), view_918, out=buf170)
        del view_918
        buf174 = reinterpret_tensor(buf56, (512, 128), (128, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf172, (512, 128), (1, 512), 0), view_916, out=buf174)
        del view_916
        buf175 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf172, buf175, 512, 128, grid=grid(512), stream=stream0)
        buf178 = buf168; del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf178, buf173, primals_365, 16384, grid=grid(16384), stream=stream0)
        del primals_365
        buf179 = reinterpret_tensor(buf172, (128, 512), (512, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf178, (128, 128), (128, 1), 0), permute_571, out=buf179)
        del permute_571
        buf180 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf178, (128, 128), (1, 128), 0), view_914, out=buf180)
        del view_914
        buf182 = reinterpret_tensor(buf179, (1, 128, 512), (65536, 512, 1), 0); del buf179  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf182, le_6, 65536, grid=grid(65536), stream=stream0)
        del le_6
        buf183 = buf173; del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf182, (128, 512), (512, 1), 0), permute_575, out=buf183)
        del permute_575
        buf188 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf178, buf183, primals_363, buf188, 16384, grid=grid(16384), stream=stream0)
        del primals_363
        buf189 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (128, 128), (128, 1), 0), permute_579, out=buf189)
        del permute_579
        buf192 = reinterpret_tensor(buf189, (1, 128, 512), (65536, 512, 1), 0); del buf189  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf192, le_7, 65536, grid=grid(65536), stream=stream0)
        del le_7
        buf193 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (128, 512), (512, 1), 0), permute_583, out=buf193)
        del permute_583
        buf181 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf186 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf187 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf191 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf196 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf197 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_340, attention_output_111, mul_181], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf178, buf183, addmm_340, buf54, primals_361, primals_362, buf188, buf193, buf181, buf186, buf187, buf191, buf196, buf197, 128, 128, grid=grid(128), stream=stream0)
        del addmm_340
        del buf178
        del primals_362
        buf184 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf182, (512, 128), (1, 512), 0), view_912, out=buf184)
        del view_912
        buf185 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf182, buf185, 512, 128, grid=grid(512), stream=stream0)
        buf190 = reinterpret_tensor(buf182, (128, 512), (512, 1), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (128, 128), (1, 128), 0), view_910, out=buf190)
        del view_910
        buf194 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (512, 128), (1, 512), 0), view_908, out=buf194)
        del view_908
        buf195 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf192, buf195, 512, 128, grid=grid(512), stream=stream0)
        buf198 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf198, buf193, primals_361, 16384, grid=grid(16384), stream=stream0)
        del primals_361
        buf199 = reinterpret_tensor(buf192, (128, 512), (512, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf198, (128, 128), (128, 1), 0), permute_587, out=buf199)
        del permute_587
        buf200 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf198, (128, 128), (1, 128), 0), view_906, out=buf200)
        del view_906
        buf202 = reinterpret_tensor(buf199, (1, 128, 512), (65536, 512, 1), 0); del buf199  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf202, le_8, 65536, grid=grid(65536), stream=stream0)
        del le_8
        buf203 = buf193; del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (128, 512), (512, 1), 0), permute_591, out=buf203)
        del permute_591
        buf208 = buf54; del buf54  # reuse
        buf233 = reinterpret_tensor(buf183, (1, 128, 128), (16384, 128, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf198, buf203, primals_359, primals_355, buf208, buf233, 16384, grid=grid(16384), stream=stream0)
        del primals_359
        buf201 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf206 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf207 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf211 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf231 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf232 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_336, layer_input_114, mul_178], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf198, buf203, addmm_336, addmm_331, primals_355, primals_356, buf208, buf201, buf206, buf207, buf211, buf231, buf232, 128, 128, grid=grid(128), stream=stream0)
        del addmm_331
        del addmm_336
        del primals_355
        del primals_356
        buf204 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (512, 128), (1, 512), 0), view_904, out=buf204)
        del view_904
        buf205 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf202, buf205, 512, 128, grid=grid(512), stream=stream0)
        buf209 = buf203; del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (128, 128), (128, 1), 0), permute_595, out=buf209)
        del permute_595
        buf210 = reinterpret_tensor(buf198, (128, 128), (128, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (128, 128), (1, 128), 0), view_902, out=buf210)
        del view_902
        # Source Nodes: [], Original ATen: []
        buf212 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf209, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_3, clone_default_4, clone_default_5, None, alias_default_3, getitem_58, getitem_59, getitem_60, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_3
        del clone_default_3
        del clone_default_4
        del clone_default_5
        del getitem_58
        del getitem_59
        del getitem_60
        buf213 = buf212[0]
        buf214 = buf212[1]
        buf215 = buf212[2]
        del buf212
        buf216 = reinterpret_tensor(buf202, (128, 512), (512, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (128, 128), (128, 1), 0), permute_608, out=buf216)
        del permute_608
        buf217 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (128, 128), (1, 128), 0), view_882, out=buf217)
        buf218 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf215, buf218, 128, 128, grid=grid(128), stream=stream0)
        buf219 = reinterpret_tensor(buf215, (128, 128), (128, 1), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf214, (128, 128), (128, 1), 0), permute_612, out=buf219)
        del permute_612
        buf220 = buf209; del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf214, (128, 128), (1, 128), 0), view_886, out=buf220)
        buf221 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf214, buf221, 128, 128, grid=grid(128), stream=stream0)
        buf222 = reinterpret_tensor(buf214, (128, 128), (128, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf213, (128, 128), (128, 1), 0), permute_616, out=buf222)
        del permute_616
        buf223 = reinterpret_tensor(buf208, (128, 128), (128, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf213, (128, 128), (1, 128), 0), view_886, out=buf223)
        del view_886
        buf224 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf213, buf224, 128, 128, grid=grid(128), stream=stream0)
        del buf213
        buf225 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf226 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf219, buf222, addmm_332, buf225, buf226, 128, 128, grid=grid(128), stream=stream0)
        del addmm_332
        buf227 = reinterpret_tensor(buf219, (1, 128, 128), (16384, 128, 1), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf227, buf222, primals_357, 16384, grid=grid(16384), stream=stream0)
        del primals_357
        buf228 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (128, 128), (128, 1), 0), permute_620, out=buf228)
        del permute_620
        buf229 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (128, 128), (1, 128), 0), view_882, out=buf229)
        buf230 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf227, buf230, 128, 128, grid=grid(128), stream=stream0)
        buf234 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (128, 128), (128, 1), 0), permute_624, out=buf234)
        del permute_624
        buf235 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (128, 128), (1, 128), 0), view_882, out=buf235)
        del view_882
        buf236 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf233, buf236, 128, 128, grid=grid(128), stream=stream0)
        buf237 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf239 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_331, mul_169, value_tensor_21], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_28.run(buf161, buf216, buf228, buf234, addmm_330, buf51, primals_337, primals_338, buf237, buf239, 512, 128, grid=grid(512), stream=stream0)
        del addmm_330
        del primals_338
        buf238 = buf161; del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf238, buf216, buf228, buf234, primals_353, 65536, grid=grid(65536), stream=stream0)
        del primals_353
        buf240 = reinterpret_tensor(buf233, (128, 128), (128, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (128, 512), (512, 1), 0), permute_628, out=buf240)
        del permute_628
        buf241 = reinterpret_tensor(buf234, (512, 128), (128, 1), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (512, 128), (1, 512), 0), view_880, out=buf241)
        del view_880
        buf242 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf238, buf242, 512, 128, grid=grid(512), stream=stream0)
        buf245 = buf227; del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf240, primals_351, buf245, 16384, grid=grid(16384), stream=stream0)
        del primals_351
        buf246 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (128, 128), (128, 1), 0), permute_632, out=buf246)
        del permute_632
        buf249 = reinterpret_tensor(buf246, (1, 128, 512), (65536, 512, 1), 0); del buf246  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf249, le_9, 65536, grid=grid(65536), stream=stream0)
        del le_9
        buf250 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (128, 512), (512, 1), 0), permute_636, out=buf250)
        del permute_636
        buf243 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf244 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf248 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf253 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf254 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_329, attention_output_108, mul_175], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf240, addmm_329, buf53, primals_349, primals_350, buf245, buf250, buf243, buf244, buf248, buf253, buf254, 128, 128, grid=grid(128), stream=stream0)
        del addmm_329
        del primals_350
        buf247 = buf216; del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (128, 128), (1, 128), 0), view_878, out=buf247)
        del view_878
        buf251 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (512, 128), (1, 512), 0), view_876, out=buf251)
        del view_876
        buf252 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf249, buf252, 512, 128, grid=grid(512), stream=stream0)
        buf255 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf255, buf250, primals_349, 16384, grid=grid(16384), stream=stream0)
        del primals_349
        buf256 = reinterpret_tensor(buf249, (128, 512), (512, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (128, 128), (128, 1), 0), permute_640, out=buf256)
        del permute_640
        buf257 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (128, 128), (1, 128), 0), view_874, out=buf257)
        del view_874
        buf259 = reinterpret_tensor(buf256, (1, 128, 512), (65536, 512, 1), 0); del buf256  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf259, le_10, 65536, grid=grid(65536), stream=stream0)
        del le_10
        buf260 = buf250; del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf259, (128, 512), (512, 1), 0), permute_644, out=buf260)
        del permute_644
        buf265 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf255, buf260, primals_347, buf265, 16384, grid=grid(16384), stream=stream0)
        del primals_347
        buf266 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf265, (128, 128), (128, 1), 0), permute_648, out=buf266)
        del permute_648
        buf269 = reinterpret_tensor(buf266, (1, 128, 512), (65536, 512, 1), 0); del buf266  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf269, le_11, 65536, grid=grid(65536), stream=stream0)
        del le_11
        buf270 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (128, 512), (512, 1), 0), permute_652, out=buf270)
        del permute_652
        buf258 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf263 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf264 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf268 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf273 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf274 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_325, attention_output_106, mul_173], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf255, buf260, addmm_325, buf52, primals_345, primals_346, buf265, buf270, buf258, buf263, buf264, buf268, buf273, buf274, 128, 128, grid=grid(128), stream=stream0)
        del addmm_325
        del buf255
        del primals_346
        buf261 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf259, (512, 128), (1, 512), 0), view_872, out=buf261)
        del view_872
        buf262 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf259, buf262, 512, 128, grid=grid(512), stream=stream0)
        buf267 = reinterpret_tensor(buf259, (128, 512), (512, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf265, (128, 128), (1, 128), 0), view_870, out=buf267)
        del view_870
        buf271 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (512, 128), (1, 512), 0), view_868, out=buf271)
        del view_868
        buf272 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf269, buf272, 512, 128, grid=grid(512), stream=stream0)
        buf275 = buf265; del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf275, buf270, primals_345, 16384, grid=grid(16384), stream=stream0)
        del primals_345
        buf276 = reinterpret_tensor(buf269, (128, 512), (512, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (128, 128), (128, 1), 0), permute_656, out=buf276)
        del permute_656
        buf277 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (128, 128), (1, 128), 0), view_866, out=buf277)
        del view_866
        buf279 = reinterpret_tensor(buf276, (1, 128, 512), (65536, 512, 1), 0); del buf276  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf279, le_12, 65536, grid=grid(65536), stream=stream0)
        del le_12
        buf280 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (128, 512), (512, 1), 0), permute_660, out=buf280)
        del permute_660
        buf285 = buf52; del buf52  # reuse
        buf310 = reinterpret_tensor(buf260, (1, 128, 128), (16384, 128, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf275, buf280, primals_343, primals_339, buf285, buf310, 16384, grid=grid(16384), stream=stream0)
        del primals_343
        buf278 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf283 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf284 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf288 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf308 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf309 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_321, layer_input_109, mul_170], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf275, buf280, addmm_321, addmm_316, primals_339, primals_340, buf285, buf278, buf283, buf284, buf288, buf308, buf309, 128, 128, grid=grid(128), stream=stream0)
        del addmm_316
        del addmm_321
        del primals_339
        del primals_340
        buf281 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (512, 128), (1, 512), 0), view_864, out=buf281)
        del view_864
        buf282 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf279, buf282, 512, 128, grid=grid(512), stream=stream0)
        buf286 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (128, 128), (128, 1), 0), permute_664, out=buf286)
        del permute_664
        buf287 = reinterpret_tensor(buf275, (128, 128), (128, 1), 0); del buf275  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (128, 128), (1, 128), 0), view_862, out=buf287)
        del view_862
        # Source Nodes: [], Original ATen: []
        buf289 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf286, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_6, clone_default_7, clone_default_8, None, alias_default_5, getitem_65, getitem_66, getitem_67, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_5
        del clone_default_6
        del clone_default_7
        del clone_default_8
        del getitem_65
        del getitem_66
        del getitem_67
        buf290 = buf289[0]
        buf291 = buf289[1]
        buf292 = buf289[2]
        del buf289
        buf293 = reinterpret_tensor(buf279, (128, 512), (512, 1), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf292, (128, 128), (128, 1), 0), permute_677, out=buf293)
        del permute_677
        buf294 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf292, (128, 128), (1, 128), 0), view_842, out=buf294)
        buf295 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf292, buf295, 128, 128, grid=grid(128), stream=stream0)
        buf296 = reinterpret_tensor(buf292, (128, 128), (128, 1), 0); del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf291, (128, 128), (128, 1), 0), permute_681, out=buf296)
        del permute_681
        buf297 = buf286; del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf291, (128, 128), (1, 128), 0), view_846, out=buf297)
        buf298 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf291, buf298, 128, 128, grid=grid(128), stream=stream0)
        buf299 = reinterpret_tensor(buf291, (128, 128), (128, 1), 0); del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf290, (128, 128), (128, 1), 0), permute_685, out=buf299)
        del permute_685
        buf300 = reinterpret_tensor(buf285, (128, 128), (128, 1), 0); del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf290, (128, 128), (1, 128), 0), view_846, out=buf300)
        del view_846
        buf301 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf290, buf301, 128, 128, grid=grid(128), stream=stream0)
        del buf290
        buf302 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf303 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf296, buf299, addmm_317, buf302, buf303, 128, 128, grid=grid(128), stream=stream0)
        del addmm_317
        buf304 = reinterpret_tensor(buf296, (1, 128, 128), (16384, 128, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf304, buf299, primals_341, 16384, grid=grid(16384), stream=stream0)
        del primals_341
        buf305 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (128, 128), (128, 1), 0), permute_689, out=buf305)
        del permute_689
        buf306 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (128, 128), (1, 128), 0), view_842, out=buf306)
        buf307 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf304, buf307, 128, 128, grid=grid(128), stream=stream0)
        buf311 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (128, 128), (128, 1), 0), permute_693, out=buf311)
        del permute_693
        buf312 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (128, 128), (1, 128), 0), view_842, out=buf312)
        del view_842
        buf313 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf310, buf313, 128, 128, grid=grid(128), stream=stream0)
        buf314 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf316 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_26.run(buf238, buf293, buf305, buf311, buf51, buf314, buf316, 512, 128, grid=grid(512), stream=stream0)
        buf315 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf315, buf293, buf305, buf311, primals_337, 65536, grid=grid(65536), stream=stream0)
        del primals_337
        buf317 = reinterpret_tensor(buf310, (128, 128), (128, 1), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (128, 512), (512, 1), 0), permute_697, out=buf317)
        del permute_697
        buf318 = reinterpret_tensor(buf311, (512, 128), (128, 1), 0); del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (512, 128), (1, 512), 0), view_840, out=buf318)
        del view_840
        buf319 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf315, buf319, 512, 128, grid=grid(512), stream=stream0)
        buf322 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf317, primals_335, buf322, 16384, grid=grid(16384), stream=stream0)
        del primals_335
        buf323 = buf305; del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf322, (128, 128), (128, 1), 0), permute_701, out=buf323)
        del permute_701
        buf326 = reinterpret_tensor(buf323, (1, 128, 512), (65536, 512, 1), 0); del buf323  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf326, le_13, 65536, grid=grid(65536), stream=stream0)
        del le_13
        buf327 = buf299; del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf326, (128, 512), (512, 1), 0), permute_705, out=buf327)
        del permute_705
        buf320 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf321 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf325 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf330 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf331 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_314, attention_output_103, mul_167], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf317, addmm_314, buf50, primals_333, primals_334, buf322, buf327, buf320, buf321, buf325, buf330, buf331, 128, 128, grid=grid(128), stream=stream0)
        del addmm_314
        del primals_334
        buf324 = buf293; del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf322, (128, 128), (1, 128), 0), view_838, out=buf324)
        del view_838
        buf328 = reinterpret_tensor(buf51, (512, 128), (128, 1), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf326, (512, 128), (1, 512), 0), view_836, out=buf328)
        del view_836
        buf329 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf326, buf329, 512, 128, grid=grid(512), stream=stream0)
        buf332 = buf322; del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf332, buf327, primals_333, 16384, grid=grid(16384), stream=stream0)
        del primals_333
        buf333 = reinterpret_tensor(buf326, (128, 512), (512, 1), 0); del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 128), (128, 1), 0), permute_709, out=buf333)
        del permute_709
        buf334 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (128, 128), (1, 128), 0), view_834, out=buf334)
        del view_834
        buf336 = reinterpret_tensor(buf333, (1, 128, 512), (65536, 512, 1), 0); del buf333  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf336, le_14, 65536, grid=grid(65536), stream=stream0)
        del le_14
        buf337 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf336, (128, 512), (512, 1), 0), permute_713, out=buf337)
        del permute_713
        buf342 = buf50; del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf332, buf337, primals_331, buf342, 16384, grid=grid(16384), stream=stream0)
        del primals_331
        buf343 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf342, (128, 128), (128, 1), 0), permute_717, out=buf343)
        del permute_717
        buf346 = reinterpret_tensor(buf343, (1, 128, 512), (65536, 512, 1), 0); del buf343  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf346, le_15, 65536, grid=grid(65536), stream=stream0)
        del le_15
        buf347 = buf317; del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf346, (128, 512), (512, 1), 0), permute_721, out=buf347)
        del permute_721
        buf335 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf340 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf341 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf345 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf350 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf351 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_310, attention_output_101, mul_165], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf332, buf337, addmm_310, buf49, primals_329, primals_330, buf342, buf347, buf335, buf340, buf341, buf345, buf350, buf351, 128, 128, grid=grid(128), stream=stream0)
        del addmm_310
        del buf332
        del primals_330
        buf338 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf336, (512, 128), (1, 512), 0), view_832, out=buf338)
        del view_832
        buf339 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf336, buf339, 512, 128, grid=grid(512), stream=stream0)
        buf344 = reinterpret_tensor(buf336, (128, 512), (512, 1), 0); del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf342, (128, 128), (1, 128), 0), view_830, out=buf344)
        del view_830
        buf348 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf346, (512, 128), (1, 512), 0), view_828, out=buf348)
        del view_828
        buf349 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf346, buf349, 512, 128, grid=grid(512), stream=stream0)
        buf352 = buf342; del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf352, buf347, primals_329, 16384, grid=grid(16384), stream=stream0)
        del primals_329
        buf353 = reinterpret_tensor(buf346, (128, 512), (512, 1), 0); del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf352, (128, 128), (128, 1), 0), permute_725, out=buf353)
        del permute_725
        buf354 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf352, (128, 128), (1, 128), 0), view_826, out=buf354)
        del view_826
        buf356 = reinterpret_tensor(buf353, (1, 128, 512), (65536, 512, 1), 0); del buf353  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf356, le_16, 65536, grid=grid(65536), stream=stream0)
        del le_16
        buf357 = buf347; del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf356, (128, 512), (512, 1), 0), permute_729, out=buf357)
        del permute_729
        buf362 = buf49; del buf49  # reuse
        buf387 = reinterpret_tensor(buf337, (1, 128, 128), (16384, 128, 1), 0); del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf352, buf357, primals_327, primals_323, buf362, buf387, 16384, grid=grid(16384), stream=stream0)
        del primals_327
        buf355 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf360 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf361 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf365 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf385 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf386 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_306, layer_input_104, mul_162], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf352, buf357, addmm_306, addmm_301, primals_323, primals_324, buf362, buf355, buf360, buf361, buf365, buf385, buf386, 128, 128, grid=grid(128), stream=stream0)
        del addmm_301
        del addmm_306
        del primals_323
        del primals_324
        buf358 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf356, (512, 128), (1, 512), 0), view_824, out=buf358)
        del view_824
        buf359 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf356, buf359, 512, 128, grid=grid(512), stream=stream0)
        buf363 = buf357; del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf362, (128, 128), (128, 1), 0), permute_733, out=buf363)
        del permute_733
        buf364 = reinterpret_tensor(buf352, (128, 128), (128, 1), 0); del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf362, (128, 128), (1, 128), 0), view_822, out=buf364)
        del view_822
        # Source Nodes: [], Original ATen: []
        buf366 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf363, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_9, clone_default_10, clone_default_11, None, alias_default_7, getitem_72, getitem_73, getitem_74, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_7
        del clone_default_10
        del clone_default_11
        del clone_default_9
        del getitem_72
        del getitem_73
        del getitem_74
        buf367 = buf366[0]
        buf368 = buf366[1]
        buf369 = buf366[2]
        del buf366
        buf370 = reinterpret_tensor(buf356, (128, 512), (512, 1), 0); del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (128, 128), (128, 1), 0), permute_746, out=buf370)
        del permute_746
        buf371 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (128, 128), (1, 128), 0), view_802, out=buf371)
        buf372 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf369, buf372, 128, 128, grid=grid(128), stream=stream0)
        buf373 = reinterpret_tensor(buf369, (128, 128), (128, 1), 0); del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf368, (128, 128), (128, 1), 0), permute_750, out=buf373)
        del permute_750
        buf374 = buf363; del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf368, (128, 128), (1, 128), 0), view_806, out=buf374)
        buf375 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf368, buf375, 128, 128, grid=grid(128), stream=stream0)
        buf376 = reinterpret_tensor(buf368, (128, 128), (128, 1), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf367, (128, 128), (128, 1), 0), permute_754, out=buf376)
        del permute_754
        buf377 = reinterpret_tensor(buf362, (128, 128), (128, 1), 0); del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf367, (128, 128), (1, 128), 0), view_806, out=buf377)
        del view_806
        buf378 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf367, buf378, 128, 128, grid=grid(128), stream=stream0)
        del buf367
        buf379 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf380 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf373, buf376, addmm_302, buf379, buf380, 128, 128, grid=grid(128), stream=stream0)
        del addmm_302
        buf381 = reinterpret_tensor(buf373, (1, 128, 128), (16384, 128, 1), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf381, buf376, primals_325, 16384, grid=grid(16384), stream=stream0)
        del primals_325
        buf382 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf381, (128, 128), (128, 1), 0), permute_758, out=buf382)
        del permute_758
        buf383 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf381, (128, 128), (1, 128), 0), view_802, out=buf383)
        buf384 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf381, buf384, 128, 128, grid=grid(128), stream=stream0)
        buf388 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (128, 128), (128, 1), 0), permute_762, out=buf388)
        del permute_762
        buf389 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (128, 128), (1, 128), 0), view_802, out=buf389)
        del view_802
        buf390 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf387, buf390, 128, 128, grid=grid(128), stream=stream0)
        buf391 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf393 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_301, mul_153, value_tensor_19], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_28.run(buf315, buf370, buf382, buf388, addmm_300, buf46, primals_305, primals_306, buf391, buf393, 512, 128, grid=grid(512), stream=stream0)
        del addmm_300
        del primals_306
        buf392 = buf315; del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf392, buf370, buf382, buf388, primals_321, 65536, grid=grid(65536), stream=stream0)
        del primals_321
        buf394 = reinterpret_tensor(buf387, (128, 128), (128, 1), 0); del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf392, (128, 512), (512, 1), 0), permute_766, out=buf394)
        del permute_766
        buf395 = reinterpret_tensor(buf388, (512, 128), (128, 1), 0); del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf392, (512, 128), (1, 512), 0), view_800, out=buf395)
        del view_800
        buf396 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf392, buf396, 512, 128, grid=grid(512), stream=stream0)
        buf399 = buf381; del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf394, primals_319, buf399, 16384, grid=grid(16384), stream=stream0)
        del primals_319
        buf400 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (128, 128), (128, 1), 0), permute_770, out=buf400)
        del permute_770
        buf403 = reinterpret_tensor(buf400, (1, 128, 512), (65536, 512, 1), 0); del buf400  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf403, le_17, 65536, grid=grid(65536), stream=stream0)
        del le_17
        buf404 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf403, (128, 512), (512, 1), 0), permute_774, out=buf404)
        del permute_774
        buf397 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf398 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf402 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf407 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf408 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_299, attention_output_98, mul_159], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf394, addmm_299, buf48, primals_317, primals_318, buf399, buf404, buf397, buf398, buf402, buf407, buf408, 128, 128, grid=grid(128), stream=stream0)
        del addmm_299
        del primals_318
        buf401 = buf370; del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (128, 128), (1, 128), 0), view_798, out=buf401)
        del view_798
        buf405 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf403, (512, 128), (1, 512), 0), view_796, out=buf405)
        del view_796
        buf406 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf403, buf406, 512, 128, grid=grid(512), stream=stream0)
        buf409 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf409, buf404, primals_317, 16384, grid=grid(16384), stream=stream0)
        del primals_317
        buf410 = reinterpret_tensor(buf403, (128, 512), (512, 1), 0); del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf409, (128, 128), (128, 1), 0), permute_778, out=buf410)
        del permute_778
        buf411 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf409, (128, 128), (1, 128), 0), view_794, out=buf411)
        del view_794
        buf413 = reinterpret_tensor(buf410, (1, 128, 512), (65536, 512, 1), 0); del buf410  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf413, le_18, 65536, grid=grid(65536), stream=stream0)
        del le_18
        buf414 = buf404; del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf413, (128, 512), (512, 1), 0), permute_782, out=buf414)
        del permute_782
        buf419 = buf48; del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf409, buf414, primals_315, buf419, 16384, grid=grid(16384), stream=stream0)
        del primals_315
        buf420 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf419, (128, 128), (128, 1), 0), permute_786, out=buf420)
        del permute_786
        buf423 = reinterpret_tensor(buf420, (1, 128, 512), (65536, 512, 1), 0); del buf420  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf423, le_19, 65536, grid=grid(65536), stream=stream0)
        del le_19
        buf424 = buf394; del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (128, 512), (512, 1), 0), permute_790, out=buf424)
        del permute_790
        buf412 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf417 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf418 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf422 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf427 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf428 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_295, attention_output_96, mul_157], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf409, buf414, addmm_295, buf47, primals_313, primals_314, buf419, buf424, buf412, buf417, buf418, buf422, buf427, buf428, 128, 128, grid=grid(128), stream=stream0)
        del addmm_295
        del buf409
        del primals_314
        buf415 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf413, (512, 128), (1, 512), 0), view_792, out=buf415)
        del view_792
        buf416 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf413, buf416, 512, 128, grid=grid(512), stream=stream0)
        buf421 = reinterpret_tensor(buf413, (128, 512), (512, 1), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf419, (128, 128), (1, 128), 0), view_790, out=buf421)
        del view_790
        buf425 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (512, 128), (1, 512), 0), view_788, out=buf425)
        del view_788
        buf426 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf423, buf426, 512, 128, grid=grid(512), stream=stream0)
        buf429 = buf419; del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf429, buf424, primals_313, 16384, grid=grid(16384), stream=stream0)
        del primals_313
        buf430 = reinterpret_tensor(buf423, (128, 512), (512, 1), 0); del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf429, (128, 128), (128, 1), 0), permute_794, out=buf430)
        del permute_794
        buf431 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf429, (128, 128), (1, 128), 0), view_786, out=buf431)
        del view_786
        buf433 = reinterpret_tensor(buf430, (1, 128, 512), (65536, 512, 1), 0); del buf430  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf433, le_20, 65536, grid=grid(65536), stream=stream0)
        del le_20
        buf434 = buf424; del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf433, (128, 512), (512, 1), 0), permute_798, out=buf434)
        del permute_798
        buf439 = buf47; del buf47  # reuse
        buf464 = reinterpret_tensor(buf414, (1, 128, 128), (16384, 128, 1), 0); del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf429, buf434, primals_311, primals_307, buf439, buf464, 16384, grid=grid(16384), stream=stream0)
        del primals_311
        buf432 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf437 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf438 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf442 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf462 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf463 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_291, layer_input_99, mul_154], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf429, buf434, addmm_291, addmm_286, primals_307, primals_308, buf439, buf432, buf437, buf438, buf442, buf462, buf463, 128, 128, grid=grid(128), stream=stream0)
        del addmm_286
        del addmm_291
        del primals_307
        del primals_308
        buf435 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf433, (512, 128), (1, 512), 0), view_784, out=buf435)
        del view_784
        buf436 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf433, buf436, 512, 128, grid=grid(512), stream=stream0)
        buf440 = buf434; del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf439, (128, 128), (128, 1), 0), permute_802, out=buf440)
        del permute_802
        buf441 = reinterpret_tensor(buf429, (128, 128), (128, 1), 0); del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf439, (128, 128), (1, 128), 0), view_782, out=buf441)
        del view_782
        # Source Nodes: [], Original ATen: []
        buf443 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf440, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_12, clone_default_13, clone_default_14, None, alias_default_9, getitem_79, getitem_80, getitem_81, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_9
        del clone_default_12
        del clone_default_13
        del clone_default_14
        del getitem_79
        del getitem_80
        del getitem_81
        buf444 = buf443[0]
        buf445 = buf443[1]
        buf446 = buf443[2]
        del buf443
        buf447 = reinterpret_tensor(buf433, (128, 512), (512, 1), 0); del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf446, (128, 128), (128, 1), 0), permute_815, out=buf447)
        del permute_815
        buf448 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf446, (128, 128), (1, 128), 0), view_762, out=buf448)
        buf449 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf446, buf449, 128, 128, grid=grid(128), stream=stream0)
        buf450 = reinterpret_tensor(buf446, (128, 128), (128, 1), 0); del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf445, (128, 128), (128, 1), 0), permute_819, out=buf450)
        del permute_819
        buf451 = buf440; del buf440  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf445, (128, 128), (1, 128), 0), view_766, out=buf451)
        buf452 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf445, buf452, 128, 128, grid=grid(128), stream=stream0)
        buf453 = reinterpret_tensor(buf445, (128, 128), (128, 1), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf444, (128, 128), (128, 1), 0), permute_823, out=buf453)
        del permute_823
        buf454 = reinterpret_tensor(buf439, (128, 128), (128, 1), 0); del buf439  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf444, (128, 128), (1, 128), 0), view_766, out=buf454)
        del view_766
        buf455 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf444, buf455, 128, 128, grid=grid(128), stream=stream0)
        del buf444
        buf456 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf457 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf450, buf453, addmm_287, buf456, buf457, 128, 128, grid=grid(128), stream=stream0)
        del addmm_287
        buf458 = reinterpret_tensor(buf450, (1, 128, 128), (16384, 128, 1), 0); del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf458, buf453, primals_309, 16384, grid=grid(16384), stream=stream0)
        del primals_309
        buf459 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf458, (128, 128), (128, 1), 0), permute_827, out=buf459)
        del permute_827
        buf460 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf458, (128, 128), (1, 128), 0), view_762, out=buf460)
        buf461 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf458, buf461, 128, 128, grid=grid(128), stream=stream0)
        buf465 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (128, 128), (128, 1), 0), permute_831, out=buf465)
        del permute_831
        buf466 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (128, 128), (1, 128), 0), view_762, out=buf466)
        del view_762
        buf467 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf464, buf467, 128, 128, grid=grid(128), stream=stream0)
        buf468 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf470 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_26.run(buf392, buf447, buf459, buf465, buf46, buf468, buf470, 512, 128, grid=grid(512), stream=stream0)
        buf469 = buf392; del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf469, buf447, buf459, buf465, primals_305, 65536, grid=grid(65536), stream=stream0)
        del primals_305
        buf471 = reinterpret_tensor(buf464, (128, 128), (128, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf469, (128, 512), (512, 1), 0), permute_835, out=buf471)
        del permute_835
        buf472 = reinterpret_tensor(buf465, (512, 128), (128, 1), 0); del buf465  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf469, (512, 128), (1, 512), 0), view_760, out=buf472)
        del view_760
        buf473 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf469, buf473, 512, 128, grid=grid(512), stream=stream0)
        buf476 = buf458; del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf471, primals_303, buf476, 16384, grid=grid(16384), stream=stream0)
        del primals_303
        buf477 = buf459; del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf476, (128, 128), (128, 1), 0), permute_839, out=buf477)
        del permute_839
        buf480 = reinterpret_tensor(buf477, (1, 128, 512), (65536, 512, 1), 0); del buf477  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf480, le_21, 65536, grid=grid(65536), stream=stream0)
        del le_21
        buf481 = buf453; del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf480, (128, 512), (512, 1), 0), permute_843, out=buf481)
        del permute_843
        buf474 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf475 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf479 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf484 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf485 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_284, attention_output_93, mul_151], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf471, addmm_284, buf45, primals_301, primals_302, buf476, buf481, buf474, buf475, buf479, buf484, buf485, 128, 128, grid=grid(128), stream=stream0)
        del addmm_284
        del primals_302
        buf478 = buf447; del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf476, (128, 128), (1, 128), 0), view_758, out=buf478)
        del view_758
        buf482 = reinterpret_tensor(buf46, (512, 128), (128, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf480, (512, 128), (1, 512), 0), view_756, out=buf482)
        del view_756
        buf483 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf480, buf483, 512, 128, grid=grid(512), stream=stream0)
        buf486 = buf476; del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf486, buf481, primals_301, 16384, grid=grid(16384), stream=stream0)
        del primals_301
        buf487 = reinterpret_tensor(buf480, (128, 512), (512, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf486, (128, 128), (128, 1), 0), permute_847, out=buf487)
        del permute_847
        buf488 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf486, (128, 128), (1, 128), 0), view_754, out=buf488)
        del view_754
        buf490 = reinterpret_tensor(buf487, (1, 128, 512), (65536, 512, 1), 0); del buf487  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf490, le_22, 65536, grid=grid(65536), stream=stream0)
        del le_22
        buf491 = buf481; del buf481  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf490, (128, 512), (512, 1), 0), permute_851, out=buf491)
        del permute_851
        buf496 = reinterpret_tensor(buf471, (1, 128, 128), (16384, 128, 1), 0); del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf486, buf491, primals_299, buf496, 16384, grid=grid(16384), stream=stream0)
        del primals_299
        buf497 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf496, (128, 128), (128, 1), 0), permute_855, out=buf497)
        del permute_855
        buf500 = reinterpret_tensor(buf497, (1, 128, 512), (65536, 512, 1), 0); del buf497  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf500, le_23, 65536, grid=grid(65536), stream=stream0)
        del le_23
        buf501 = reinterpret_tensor(buf45, (128, 128), (128, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf500, (128, 512), (512, 1), 0), permute_859, out=buf501)
        del permute_859
        buf489 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf494 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf495 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf499 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf504 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf505 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_280, attention_output_91, mul_149], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf486, buf491, addmm_280, buf44, primals_297, primals_298, buf496, buf501, buf489, buf494, buf495, buf499, buf504, buf505, 128, 128, grid=grid(128), stream=stream0)
        del addmm_280
        del buf44
        del primals_298
        buf492 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf490, (512, 128), (1, 512), 0), view_752, out=buf492)
        del view_752
        buf493 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf490, buf493, 512, 128, grid=grid(512), stream=stream0)
        buf498 = reinterpret_tensor(buf490, (128, 512), (512, 1), 0); del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf496, (128, 128), (1, 128), 0), view_750, out=buf498)
        del view_750
        buf502 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf500, (512, 128), (1, 512), 0), view_748, out=buf502)
        del view_748
        buf503 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf500, buf503, 512, 128, grid=grid(512), stream=stream0)
        buf506 = buf496; del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf506, buf501, primals_297, 16384, grid=grid(16384), stream=stream0)
        del primals_297
        buf507 = reinterpret_tensor(buf500, (128, 512), (512, 1), 0); del buf500  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf506, (128, 128), (128, 1), 0), permute_863, out=buf507)
        del permute_863
        buf508 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf506, (128, 128), (1, 128), 0), view_746, out=buf508)
        del view_746
        buf510 = reinterpret_tensor(buf507, (1, 128, 512), (65536, 512, 1), 0); del buf507  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf510, le_24, 65536, grid=grid(65536), stream=stream0)
        del le_24
        buf511 = buf501; del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf510, (128, 512), (512, 1), 0), permute_867, out=buf511)
        del permute_867
        buf516 = reinterpret_tensor(buf491, (1, 128, 128), (16384, 128, 1), 0); del buf491  # reuse
        buf541 = buf486; del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf506, buf511, primals_295, primals_291, buf516, buf541, 16384, grid=grid(16384), stream=stream0)
        del primals_295
        buf509 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf514 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf515 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf519 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf539 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf540 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_276, layer_input_94, mul_146], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf506, buf511, addmm_276, addmm_271, primals_291, primals_292, buf516, buf509, buf514, buf515, buf519, buf539, buf540, 128, 128, grid=grid(128), stream=stream0)
        del addmm_271
        del addmm_276
        del primals_291
        del primals_292
        buf512 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf510, (512, 128), (1, 512), 0), view_744, out=buf512)
        del view_744
        buf513 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf510, buf513, 512, 128, grid=grid(512), stream=stream0)
        buf517 = buf511; del buf511  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf516, (128, 128), (128, 1), 0), permute_871, out=buf517)
        del permute_871
        buf518 = reinterpret_tensor(buf506, (128, 128), (128, 1), 0); del buf506  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf516, (128, 128), (1, 128), 0), view_742, out=buf518)
        del view_742
        # Source Nodes: [], Original ATen: []
        buf520 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf517, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_15, clone_default_16, clone_default_17, None, alias_default_11, getitem_86, getitem_87, getitem_88, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_11
        del clone_default_15
        del clone_default_16
        del clone_default_17
        del getitem_86
        del getitem_87
        del getitem_88
        buf521 = buf520[0]
        buf522 = buf520[1]
        buf523 = buf520[2]
        del buf520
        buf524 = reinterpret_tensor(buf510, (128, 512), (512, 1), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf523, (128, 128), (128, 1), 0), permute_884, out=buf524)
        del permute_884
        buf525 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf523, (128, 128), (1, 128), 0), view_722, out=buf525)
        buf526 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf523, buf526, 128, 128, grid=grid(128), stream=stream0)
        buf527 = reinterpret_tensor(buf523, (128, 128), (128, 1), 0); del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf522, (128, 128), (128, 1), 0), permute_888, out=buf527)
        del permute_888
        buf528 = buf517; del buf517  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf522, (128, 128), (1, 128), 0), view_726, out=buf528)
        buf529 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf522, buf529, 128, 128, grid=grid(128), stream=stream0)
        buf530 = reinterpret_tensor(buf522, (128, 128), (128, 1), 0); del buf522  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf521, (128, 128), (128, 1), 0), permute_892, out=buf530)
        del permute_892
        buf531 = reinterpret_tensor(buf516, (128, 128), (128, 1), 0); del buf516  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf521, (128, 128), (1, 128), 0), view_726, out=buf531)
        del view_726
        buf532 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf521, buf532, 128, 128, grid=grid(128), stream=stream0)
        del buf521
        buf533 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf534 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf527, buf530, addmm_272, buf533, buf534, 128, 128, grid=grid(128), stream=stream0)
        del addmm_272
        buf535 = reinterpret_tensor(buf527, (1, 128, 128), (16384, 128, 1), 0); del buf527  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf535, buf530, primals_293, 16384, grid=grid(16384), stream=stream0)
        del primals_293
        buf536 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf535, (128, 128), (128, 1), 0), permute_896, out=buf536)
        del permute_896
        buf537 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf535, (128, 128), (1, 128), 0), view_722, out=buf537)
        buf538 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf535, buf538, 128, 128, grid=grid(128), stream=stream0)
        buf542 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf541, (128, 128), (128, 1), 0), permute_900, out=buf542)
        del permute_900
        buf543 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf541, (128, 128), (1, 128), 0), view_722, out=buf543)
        del view_722
        buf544 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf541, buf544, 128, 128, grid=grid(128), stream=stream0)
        buf545 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf547 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_271, mul_137, value_tensor_17], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_28.run(buf469, buf524, buf536, buf542, addmm_270, buf41, primals_273, primals_274, buf545, buf547, 512, 128, grid=grid(512), stream=stream0)
        del addmm_270
        del primals_274
        buf546 = buf469; del buf469  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf546, buf524, buf536, buf542, primals_289, 65536, grid=grid(65536), stream=stream0)
        del primals_289
        buf548 = reinterpret_tensor(buf541, (128, 128), (128, 1), 0); del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf546, (128, 512), (512, 1), 0), permute_904, out=buf548)
        del permute_904
        buf549 = reinterpret_tensor(buf542, (512, 128), (128, 1), 0); del buf542  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf546, (512, 128), (1, 512), 0), view_720, out=buf549)
        del view_720
        buf550 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf546, buf550, 512, 128, grid=grid(512), stream=stream0)
        buf553 = buf535; del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf548, primals_287, buf553, 16384, grid=grid(16384), stream=stream0)
        del primals_287
        buf554 = buf536; del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf553, (128, 128), (128, 1), 0), permute_908, out=buf554)
        del permute_908
        buf557 = reinterpret_tensor(buf554, (1, 128, 512), (65536, 512, 1), 0); del buf554  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf557, le_25, 65536, grid=grid(65536), stream=stream0)
        del le_25
        buf558 = buf530; del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf557, (128, 512), (512, 1), 0), permute_912, out=buf558)
        del permute_912
        buf551 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf552 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf556 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf561 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf562 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_269, attention_output_88, mul_143], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf548, addmm_269, buf43, primals_285, primals_286, buf553, buf558, buf551, buf552, buf556, buf561, buf562, 128, 128, grid=grid(128), stream=stream0)
        del addmm_269
        del primals_286
        buf555 = buf524; del buf524  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf553, (128, 128), (1, 128), 0), view_718, out=buf555)
        del view_718
        buf559 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf557, (512, 128), (1, 512), 0), view_716, out=buf559)
        del view_716
        buf560 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf557, buf560, 512, 128, grid=grid(512), stream=stream0)
        buf563 = buf553; del buf553  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf563, buf558, primals_285, 16384, grid=grid(16384), stream=stream0)
        del primals_285
        buf564 = reinterpret_tensor(buf557, (128, 512), (512, 1), 0); del buf557  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf563, (128, 128), (128, 1), 0), permute_916, out=buf564)
        del permute_916
        buf565 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf563, (128, 128), (1, 128), 0), view_714, out=buf565)
        del view_714
        buf567 = reinterpret_tensor(buf564, (1, 128, 512), (65536, 512, 1), 0); del buf564  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf567, le_26, 65536, grid=grid(65536), stream=stream0)
        del le_26
        buf568 = buf558; del buf558  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf567, (128, 512), (512, 1), 0), permute_920, out=buf568)
        del permute_920
        buf573 = reinterpret_tensor(buf548, (1, 128, 128), (16384, 128, 1), 0); del buf548  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf563, buf568, primals_283, buf573, 16384, grid=grid(16384), stream=stream0)
        del primals_283
        buf574 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf573, (128, 128), (128, 1), 0), permute_924, out=buf574)
        del permute_924
        buf577 = reinterpret_tensor(buf574, (1, 128, 512), (65536, 512, 1), 0); del buf574  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf577, le_27, 65536, grid=grid(65536), stream=stream0)
        del le_27
        buf578 = reinterpret_tensor(buf43, (128, 128), (128, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf577, (128, 512), (512, 1), 0), permute_928, out=buf578)
        del permute_928
        buf566 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf571 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf572 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf576 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf581 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf582 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_265, attention_output_86, mul_141], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf563, buf568, addmm_265, buf42, primals_281, primals_282, buf573, buf578, buf566, buf571, buf572, buf576, buf581, buf582, 128, 128, grid=grid(128), stream=stream0)
        del addmm_265
        del buf42
        del primals_282
        buf569 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf567, (512, 128), (1, 512), 0), view_712, out=buf569)
        del view_712
        buf570 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf567, buf570, 512, 128, grid=grid(512), stream=stream0)
        buf575 = reinterpret_tensor(buf567, (128, 512), (512, 1), 0); del buf567  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf573, (128, 128), (1, 128), 0), view_710, out=buf575)
        del view_710
        buf579 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf577, (512, 128), (1, 512), 0), view_708, out=buf579)
        del view_708
        buf580 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf577, buf580, 512, 128, grid=grid(512), stream=stream0)
        buf583 = buf573; del buf573  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf583, buf578, primals_281, 16384, grid=grid(16384), stream=stream0)
        del primals_281
        buf584 = reinterpret_tensor(buf577, (128, 512), (512, 1), 0); del buf577  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf583, (128, 128), (128, 1), 0), permute_932, out=buf584)
        del permute_932
        buf585 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf583, (128, 128), (1, 128), 0), view_706, out=buf585)
        del view_706
        buf587 = reinterpret_tensor(buf584, (1, 128, 512), (65536, 512, 1), 0); del buf584  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf587, le_28, 65536, grid=grid(65536), stream=stream0)
        del le_28
        buf588 = buf578; del buf578  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf587, (128, 512), (512, 1), 0), permute_936, out=buf588)
        del permute_936
        buf593 = reinterpret_tensor(buf568, (1, 128, 128), (16384, 128, 1), 0); del buf568  # reuse
        buf618 = buf563; del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf583, buf588, primals_279, primals_275, buf593, buf618, 16384, grid=grid(16384), stream=stream0)
        del primals_279
        buf586 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf591 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf592 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf596 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf616 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf617 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_261, layer_input_89, mul_138], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf583, buf588, addmm_261, addmm_256, primals_275, primals_276, buf593, buf586, buf591, buf592, buf596, buf616, buf617, 128, 128, grid=grid(128), stream=stream0)
        del addmm_256
        del addmm_261
        del primals_275
        del primals_276
        buf589 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf587, (512, 128), (1, 512), 0), view_704, out=buf589)
        del view_704
        buf590 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf587, buf590, 512, 128, grid=grid(512), stream=stream0)
        buf594 = buf588; del buf588  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf593, (128, 128), (128, 1), 0), permute_940, out=buf594)
        del permute_940
        buf595 = reinterpret_tensor(buf583, (128, 128), (128, 1), 0); del buf583  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf593, (128, 128), (1, 128), 0), view_702, out=buf595)
        del view_702
        # Source Nodes: [], Original ATen: []
        buf597 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf594, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_18, clone_default_19, clone_default_20, None, alias_default_13, getitem_93, getitem_94, getitem_95, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_13
        del clone_default_18
        del clone_default_19
        del clone_default_20
        del getitem_93
        del getitem_94
        del getitem_95
        buf598 = buf597[0]
        buf599 = buf597[1]
        buf600 = buf597[2]
        del buf597
        buf601 = reinterpret_tensor(buf587, (128, 512), (512, 1), 0); del buf587  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf600, (128, 128), (128, 1), 0), permute_953, out=buf601)
        del permute_953
        buf602 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf600, (128, 128), (1, 128), 0), view_682, out=buf602)
        buf603 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf600, buf603, 128, 128, grid=grid(128), stream=stream0)
        buf604 = reinterpret_tensor(buf600, (128, 128), (128, 1), 0); del buf600  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf599, (128, 128), (128, 1), 0), permute_957, out=buf604)
        del permute_957
        buf605 = buf594; del buf594  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf599, (128, 128), (1, 128), 0), view_686, out=buf605)
        buf606 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf599, buf606, 128, 128, grid=grid(128), stream=stream0)
        buf607 = reinterpret_tensor(buf599, (128, 128), (128, 1), 0); del buf599  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf598, (128, 128), (128, 1), 0), permute_961, out=buf607)
        del permute_961
        buf608 = reinterpret_tensor(buf593, (128, 128), (128, 1), 0); del buf593  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf598, (128, 128), (1, 128), 0), view_686, out=buf608)
        del view_686
        buf609 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf598, buf609, 128, 128, grid=grid(128), stream=stream0)
        del buf598
        buf610 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf611 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf604, buf607, addmm_257, buf610, buf611, 128, 128, grid=grid(128), stream=stream0)
        del addmm_257
        buf612 = reinterpret_tensor(buf604, (1, 128, 128), (16384, 128, 1), 0); del buf604  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf612, buf607, primals_277, 16384, grid=grid(16384), stream=stream0)
        del primals_277
        buf613 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf612, (128, 128), (128, 1), 0), permute_965, out=buf613)
        del permute_965
        buf614 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf612, (128, 128), (1, 128), 0), view_682, out=buf614)
        buf615 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf612, buf615, 128, 128, grid=grid(128), stream=stream0)
        buf619 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf618, (128, 128), (128, 1), 0), permute_969, out=buf619)
        del permute_969
        buf620 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf618, (128, 128), (1, 128), 0), view_682, out=buf620)
        del view_682
        buf621 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf618, buf621, 128, 128, grid=grid(128), stream=stream0)
        buf622 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf624 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_26.run(buf546, buf601, buf613, buf619, buf41, buf622, buf624, 512, 128, grid=grid(512), stream=stream0)
        buf623 = buf546; del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf623, buf601, buf613, buf619, primals_273, 65536, grid=grid(65536), stream=stream0)
        del primals_273
        buf625 = reinterpret_tensor(buf618, (128, 128), (128, 1), 0); del buf618  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf623, (128, 512), (512, 1), 0), permute_973, out=buf625)
        del permute_973
        buf626 = reinterpret_tensor(buf619, (512, 128), (128, 1), 0); del buf619  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf623, (512, 128), (1, 512), 0), view_680, out=buf626)
        del view_680
        buf627 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf623, buf627, 512, 128, grid=grid(512), stream=stream0)
        buf630 = buf612; del buf612  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf625, primals_271, buf630, 16384, grid=grid(16384), stream=stream0)
        del primals_271
        buf631 = buf613; del buf613  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf630, (128, 128), (128, 1), 0), permute_977, out=buf631)
        del permute_977
        buf634 = reinterpret_tensor(buf631, (1, 128, 512), (65536, 512, 1), 0); del buf631  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf634, le_29, 65536, grid=grid(65536), stream=stream0)
        del le_29
        buf635 = buf607; del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf634, (128, 512), (512, 1), 0), permute_981, out=buf635)
        del permute_981
        buf628 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf629 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf633 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf638 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf639 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_254, attention_output_83, mul_135], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf625, addmm_254, buf40, primals_269, primals_270, buf630, buf635, buf628, buf629, buf633, buf638, buf639, 128, 128, grid=grid(128), stream=stream0)
        del addmm_254
        del primals_270
        buf632 = buf601; del buf601  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf630, (128, 128), (1, 128), 0), view_678, out=buf632)
        del view_678
        buf636 = reinterpret_tensor(buf41, (512, 128), (128, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf634, (512, 128), (1, 512), 0), view_676, out=buf636)
        del view_676
        buf637 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf634, buf637, 512, 128, grid=grid(512), stream=stream0)
        buf640 = buf630; del buf630  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf640, buf635, primals_269, 16384, grid=grid(16384), stream=stream0)
        del primals_269
        buf641 = reinterpret_tensor(buf634, (128, 512), (512, 1), 0); del buf634  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf640, (128, 128), (128, 1), 0), permute_985, out=buf641)
        del permute_985
        buf642 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf640, (128, 128), (1, 128), 0), view_674, out=buf642)
        del view_674
        buf644 = reinterpret_tensor(buf641, (1, 128, 512), (65536, 512, 1), 0); del buf641  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf644, le_30, 65536, grid=grid(65536), stream=stream0)
        del le_30
        buf645 = buf635; del buf635  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf644, (128, 512), (512, 1), 0), permute_989, out=buf645)
        del permute_989
        buf650 = reinterpret_tensor(buf625, (1, 128, 128), (16384, 128, 1), 0); del buf625  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf640, buf645, primals_267, buf650, 16384, grid=grid(16384), stream=stream0)
        del primals_267
        buf651 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf650, (128, 128), (128, 1), 0), permute_993, out=buf651)
        del permute_993
        buf654 = reinterpret_tensor(buf651, (1, 128, 512), (65536, 512, 1), 0); del buf651  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf654, le_31, 65536, grid=grid(65536), stream=stream0)
        del le_31
        buf655 = reinterpret_tensor(buf40, (128, 128), (128, 1), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf654, (128, 512), (512, 1), 0), permute_997, out=buf655)
        del permute_997
        buf643 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf648 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf649 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf653 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf658 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf659 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_250, attention_output_81, mul_133], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf640, buf645, addmm_250, buf39, primals_265, primals_266, buf650, buf655, buf643, buf648, buf649, buf653, buf658, buf659, 128, 128, grid=grid(128), stream=stream0)
        del addmm_250
        del buf39
        del primals_266
        buf646 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf644, (512, 128), (1, 512), 0), view_672, out=buf646)
        del view_672
        buf647 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf644, buf647, 512, 128, grid=grid(512), stream=stream0)
        buf652 = reinterpret_tensor(buf644, (128, 512), (512, 1), 0); del buf644  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf650, (128, 128), (1, 128), 0), view_670, out=buf652)
        del view_670
        buf656 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf654, (512, 128), (1, 512), 0), view_668, out=buf656)
        del view_668
        buf657 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf654, buf657, 512, 128, grid=grid(512), stream=stream0)
        buf660 = buf650; del buf650  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf660, buf655, primals_265, 16384, grid=grid(16384), stream=stream0)
        del primals_265
        buf661 = reinterpret_tensor(buf654, (128, 512), (512, 1), 0); del buf654  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf660, (128, 128), (128, 1), 0), permute_1001, out=buf661)
        del permute_1001
        buf662 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf660, (128, 128), (1, 128), 0), view_666, out=buf662)
        del view_666
        buf664 = reinterpret_tensor(buf661, (1, 128, 512), (65536, 512, 1), 0); del buf661  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf664, le_32, 65536, grid=grid(65536), stream=stream0)
        del le_32
        buf665 = buf655; del buf655  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf664, (128, 512), (512, 1), 0), permute_1005, out=buf665)
        del permute_1005
        buf670 = reinterpret_tensor(buf645, (1, 128, 128), (16384, 128, 1), 0); del buf645  # reuse
        buf695 = buf640; del buf640  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf660, buf665, primals_263, primals_259, buf670, buf695, 16384, grid=grid(16384), stream=stream0)
        del primals_263
        buf663 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf668 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf669 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf673 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf693 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf694 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_246, layer_input_84, mul_130], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf660, buf665, addmm_246, addmm_241, primals_259, primals_260, buf670, buf663, buf668, buf669, buf673, buf693, buf694, 128, 128, grid=grid(128), stream=stream0)
        del addmm_241
        del addmm_246
        del primals_259
        del primals_260
        buf666 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf664, (512, 128), (1, 512), 0), view_664, out=buf666)
        del view_664
        buf667 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf664, buf667, 512, 128, grid=grid(512), stream=stream0)
        buf671 = buf665; del buf665  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf670, (128, 128), (128, 1), 0), permute_1009, out=buf671)
        del permute_1009
        buf672 = reinterpret_tensor(buf660, (128, 128), (128, 1), 0); del buf660  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf670, (128, 128), (1, 128), 0), view_662, out=buf672)
        del view_662
        # Source Nodes: [], Original ATen: []
        buf674 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf671, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_21, clone_default_22, clone_default_23, None, alias_default_15, getitem_100, getitem_101, getitem_102, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_15
        del clone_default_21
        del clone_default_22
        del clone_default_23
        del getitem_100
        del getitem_101
        del getitem_102
        buf675 = buf674[0]
        buf676 = buf674[1]
        buf677 = buf674[2]
        del buf674
        buf678 = reinterpret_tensor(buf664, (128, 512), (512, 1), 0); del buf664  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf677, (128, 128), (128, 1), 0), permute_1022, out=buf678)
        del permute_1022
        buf679 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf677, (128, 128), (1, 128), 0), view_642, out=buf679)
        buf680 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf677, buf680, 128, 128, grid=grid(128), stream=stream0)
        buf681 = reinterpret_tensor(buf677, (128, 128), (128, 1), 0); del buf677  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf676, (128, 128), (128, 1), 0), permute_1026, out=buf681)
        del permute_1026
        buf682 = buf671; del buf671  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf676, (128, 128), (1, 128), 0), view_646, out=buf682)
        buf683 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf676, buf683, 128, 128, grid=grid(128), stream=stream0)
        buf684 = reinterpret_tensor(buf676, (128, 128), (128, 1), 0); del buf676  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf675, (128, 128), (128, 1), 0), permute_1030, out=buf684)
        del permute_1030
        buf685 = reinterpret_tensor(buf670, (128, 128), (128, 1), 0); del buf670  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf675, (128, 128), (1, 128), 0), view_646, out=buf685)
        del view_646
        buf686 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf675, buf686, 128, 128, grid=grid(128), stream=stream0)
        del buf675
        buf687 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf688 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf681, buf684, addmm_242, buf687, buf688, 128, 128, grid=grid(128), stream=stream0)
        del addmm_242
        buf689 = reinterpret_tensor(buf681, (1, 128, 128), (16384, 128, 1), 0); del buf681  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf689, buf684, primals_261, 16384, grid=grid(16384), stream=stream0)
        del primals_261
        buf690 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf689, (128, 128), (128, 1), 0), permute_1034, out=buf690)
        del permute_1034
        buf691 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf689, (128, 128), (1, 128), 0), view_642, out=buf691)
        buf692 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf689, buf692, 128, 128, grid=grid(128), stream=stream0)
        buf696 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf695, (128, 128), (128, 1), 0), permute_1038, out=buf696)
        del permute_1038
        buf697 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf695, (128, 128), (1, 128), 0), view_642, out=buf697)
        del view_642
        buf698 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf695, buf698, 128, 128, grid=grid(128), stream=stream0)
        buf699 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf701 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_241, mul_121, value_tensor_15], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_28.run(buf623, buf678, buf690, buf696, addmm_240, buf36, primals_241, primals_242, buf699, buf701, 512, 128, grid=grid(512), stream=stream0)
        del addmm_240
        del primals_242
        buf700 = buf623; del buf623  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf700, buf678, buf690, buf696, primals_257, 65536, grid=grid(65536), stream=stream0)
        del primals_257
        buf702 = reinterpret_tensor(buf695, (128, 128), (128, 1), 0); del buf695  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf700, (128, 512), (512, 1), 0), permute_1042, out=buf702)
        del permute_1042
        buf703 = reinterpret_tensor(buf696, (512, 128), (128, 1), 0); del buf696  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf700, (512, 128), (1, 512), 0), view_640, out=buf703)
        del view_640
        buf704 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf700, buf704, 512, 128, grid=grid(512), stream=stream0)
        buf707 = buf689; del buf689  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf702, primals_255, buf707, 16384, grid=grid(16384), stream=stream0)
        del primals_255
        buf708 = buf690; del buf690  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf707, (128, 128), (128, 1), 0), permute_1046, out=buf708)
        del permute_1046
        buf711 = reinterpret_tensor(buf708, (1, 128, 512), (65536, 512, 1), 0); del buf708  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf711, le_33, 65536, grid=grid(65536), stream=stream0)
        del le_33
        buf712 = buf684; del buf684  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf711, (128, 512), (512, 1), 0), permute_1050, out=buf712)
        del permute_1050
        buf705 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf706 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf710 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf715 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf716 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_239, attention_output_78, mul_127], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf702, addmm_239, buf38, primals_253, primals_254, buf707, buf712, buf705, buf706, buf710, buf715, buf716, 128, 128, grid=grid(128), stream=stream0)
        del addmm_239
        del primals_254
        buf709 = buf678; del buf678  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf707, (128, 128), (1, 128), 0), view_638, out=buf709)
        del view_638
        buf713 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf711, (512, 128), (1, 512), 0), view_636, out=buf713)
        del view_636
        buf714 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf711, buf714, 512, 128, grid=grid(512), stream=stream0)
        buf717 = buf707; del buf707  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf717, buf712, primals_253, 16384, grid=grid(16384), stream=stream0)
        del primals_253
        buf718 = reinterpret_tensor(buf711, (128, 512), (512, 1), 0); del buf711  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf717, (128, 128), (128, 1), 0), permute_1054, out=buf718)
        del permute_1054
        buf719 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf717, (128, 128), (1, 128), 0), view_634, out=buf719)
        del view_634
        buf721 = reinterpret_tensor(buf718, (1, 128, 512), (65536, 512, 1), 0); del buf718  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf721, le_34, 65536, grid=grid(65536), stream=stream0)
        del le_34
        buf722 = buf712; del buf712  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf721, (128, 512), (512, 1), 0), permute_1058, out=buf722)
        del permute_1058
        buf727 = reinterpret_tensor(buf702, (1, 128, 128), (16384, 128, 1), 0); del buf702  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf717, buf722, primals_251, buf727, 16384, grid=grid(16384), stream=stream0)
        del primals_251
        buf728 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf727, (128, 128), (128, 1), 0), permute_1062, out=buf728)
        del permute_1062
        buf731 = reinterpret_tensor(buf728, (1, 128, 512), (65536, 512, 1), 0); del buf728  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf731, le_35, 65536, grid=grid(65536), stream=stream0)
        del le_35
        buf732 = reinterpret_tensor(buf38, (128, 128), (128, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf731, (128, 512), (512, 1), 0), permute_1066, out=buf732)
        del permute_1066
        buf720 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf725 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf726 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf730 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf735 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf736 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_235, attention_output_76, mul_125], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf717, buf722, addmm_235, buf37, primals_249, primals_250, buf727, buf732, buf720, buf725, buf726, buf730, buf735, buf736, 128, 128, grid=grid(128), stream=stream0)
        del addmm_235
        del buf37
        del primals_250
        buf723 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf721, (512, 128), (1, 512), 0), view_632, out=buf723)
        del view_632
        buf724 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf721, buf724, 512, 128, grid=grid(512), stream=stream0)
        buf729 = reinterpret_tensor(buf721, (128, 512), (512, 1), 0); del buf721  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf727, (128, 128), (1, 128), 0), view_630, out=buf729)
        del view_630
        buf733 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf731, (512, 128), (1, 512), 0), view_628, out=buf733)
        del view_628
        buf734 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf731, buf734, 512, 128, grid=grid(512), stream=stream0)
        buf737 = buf727; del buf727  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf737, buf732, primals_249, 16384, grid=grid(16384), stream=stream0)
        del primals_249
        buf738 = reinterpret_tensor(buf731, (128, 512), (512, 1), 0); del buf731  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf737, (128, 128), (128, 1), 0), permute_1070, out=buf738)
        del permute_1070
        buf739 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf737, (128, 128), (1, 128), 0), view_626, out=buf739)
        del view_626
        buf741 = reinterpret_tensor(buf738, (1, 128, 512), (65536, 512, 1), 0); del buf738  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf741, le_36, 65536, grid=grid(65536), stream=stream0)
        del le_36
        buf742 = buf732; del buf732  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf741, (128, 512), (512, 1), 0), permute_1074, out=buf742)
        del permute_1074
        buf747 = reinterpret_tensor(buf722, (1, 128, 128), (16384, 128, 1), 0); del buf722  # reuse
        buf772 = buf717; del buf717  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf737, buf742, primals_247, primals_243, buf747, buf772, 16384, grid=grid(16384), stream=stream0)
        del primals_247
        buf740 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf745 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf746 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf750 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf770 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf771 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_231, layer_input_79, mul_122], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf737, buf742, addmm_231, addmm_226, primals_243, primals_244, buf747, buf740, buf745, buf746, buf750, buf770, buf771, 128, 128, grid=grid(128), stream=stream0)
        del addmm_226
        del addmm_231
        del primals_243
        del primals_244
        buf743 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf741, (512, 128), (1, 512), 0), view_624, out=buf743)
        del view_624
        buf744 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf741, buf744, 512, 128, grid=grid(512), stream=stream0)
        buf748 = buf742; del buf742  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf747, (128, 128), (128, 1), 0), permute_1078, out=buf748)
        del permute_1078
        buf749 = reinterpret_tensor(buf737, (128, 128), (128, 1), 0); del buf737  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf747, (128, 128), (1, 128), 0), view_622, out=buf749)
        del view_622
        # Source Nodes: [], Original ATen: []
        buf751 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf748, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_24, clone_default_25, clone_default_26, None, alias_default_17, getitem_107, getitem_108, getitem_109, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_17
        del clone_default_24
        del clone_default_25
        del clone_default_26
        del getitem_107
        del getitem_108
        del getitem_109
        buf752 = buf751[0]
        buf753 = buf751[1]
        buf754 = buf751[2]
        del buf751
        buf755 = reinterpret_tensor(buf741, (128, 512), (512, 1), 0); del buf741  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf754, (128, 128), (128, 1), 0), permute_1091, out=buf755)
        del permute_1091
        buf756 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf754, (128, 128), (1, 128), 0), view_602, out=buf756)
        buf757 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf754, buf757, 128, 128, grid=grid(128), stream=stream0)
        buf758 = reinterpret_tensor(buf754, (128, 128), (128, 1), 0); del buf754  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf753, (128, 128), (128, 1), 0), permute_1095, out=buf758)
        del permute_1095
        buf759 = buf748; del buf748  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf753, (128, 128), (1, 128), 0), view_606, out=buf759)
        buf760 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf753, buf760, 128, 128, grid=grid(128), stream=stream0)
        buf761 = reinterpret_tensor(buf753, (128, 128), (128, 1), 0); del buf753  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf752, (128, 128), (128, 1), 0), permute_1099, out=buf761)
        del permute_1099
        buf762 = reinterpret_tensor(buf747, (128, 128), (128, 1), 0); del buf747  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf752, (128, 128), (1, 128), 0), view_606, out=buf762)
        del view_606
        buf763 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf752, buf763, 128, 128, grid=grid(128), stream=stream0)
        del buf752
        buf764 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf765 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf758, buf761, addmm_227, buf764, buf765, 128, 128, grid=grid(128), stream=stream0)
        del addmm_227
        buf766 = reinterpret_tensor(buf758, (1, 128, 128), (16384, 128, 1), 0); del buf758  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf766, buf761, primals_245, 16384, grid=grid(16384), stream=stream0)
        del primals_245
        buf767 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf766, (128, 128), (128, 1), 0), permute_1103, out=buf767)
        del permute_1103
        buf768 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf766, (128, 128), (1, 128), 0), view_602, out=buf768)
        buf769 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf766, buf769, 128, 128, grid=grid(128), stream=stream0)
        buf773 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf772, (128, 128), (128, 1), 0), permute_1107, out=buf773)
        del permute_1107
        buf774 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf772, (128, 128), (1, 128), 0), view_602, out=buf774)
        del view_602
        buf775 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf772, buf775, 128, 128, grid=grid(128), stream=stream0)
        buf776 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf778 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_26.run(buf700, buf755, buf767, buf773, buf36, buf776, buf778, 512, 128, grid=grid(512), stream=stream0)
        buf777 = buf700; del buf700  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf777, buf755, buf767, buf773, primals_241, 65536, grid=grid(65536), stream=stream0)
        del primals_241
        buf779 = reinterpret_tensor(buf772, (128, 128), (128, 1), 0); del buf772  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf777, (128, 512), (512, 1), 0), permute_1111, out=buf779)
        del permute_1111
        buf780 = reinterpret_tensor(buf773, (512, 128), (128, 1), 0); del buf773  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf777, (512, 128), (1, 512), 0), view_600, out=buf780)
        del view_600
        buf781 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf777, buf781, 512, 128, grid=grid(512), stream=stream0)
        buf784 = buf766; del buf766  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf779, primals_239, buf784, 16384, grid=grid(16384), stream=stream0)
        del primals_239
        buf785 = buf767; del buf767  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf784, (128, 128), (128, 1), 0), permute_1115, out=buf785)
        del permute_1115
        buf788 = reinterpret_tensor(buf785, (1, 128, 512), (65536, 512, 1), 0); del buf785  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf788, le_37, 65536, grid=grid(65536), stream=stream0)
        del le_37
        buf789 = buf761; del buf761  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf788, (128, 512), (512, 1), 0), permute_1119, out=buf789)
        del permute_1119
        buf782 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf783 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf787 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf792 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf793 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_224, attention_output_73, mul_119], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf779, addmm_224, buf35, primals_237, primals_238, buf784, buf789, buf782, buf783, buf787, buf792, buf793, 128, 128, grid=grid(128), stream=stream0)
        del addmm_224
        del primals_238
        buf786 = buf755; del buf755  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf784, (128, 128), (1, 128), 0), view_598, out=buf786)
        del view_598
        buf790 = reinterpret_tensor(buf36, (512, 128), (128, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf788, (512, 128), (1, 512), 0), view_596, out=buf790)
        del view_596
        buf791 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf788, buf791, 512, 128, grid=grid(512), stream=stream0)
        buf794 = buf784; del buf784  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf794, buf789, primals_237, 16384, grid=grid(16384), stream=stream0)
        del primals_237
        buf795 = reinterpret_tensor(buf788, (128, 512), (512, 1), 0); del buf788  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf794, (128, 128), (128, 1), 0), permute_1123, out=buf795)
        del permute_1123
        buf796 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf794, (128, 128), (1, 128), 0), view_594, out=buf796)
        del view_594
        buf798 = reinterpret_tensor(buf795, (1, 128, 512), (65536, 512, 1), 0); del buf795  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf798, le_38, 65536, grid=grid(65536), stream=stream0)
        del le_38
        buf799 = buf789; del buf789  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf798, (128, 512), (512, 1), 0), permute_1127, out=buf799)
        del permute_1127
        buf804 = reinterpret_tensor(buf779, (1, 128, 128), (16384, 128, 1), 0); del buf779  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf794, buf799, primals_235, buf804, 16384, grid=grid(16384), stream=stream0)
        del primals_235
        buf805 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf804, (128, 128), (128, 1), 0), permute_1131, out=buf805)
        del permute_1131
        buf808 = reinterpret_tensor(buf805, (1, 128, 512), (65536, 512, 1), 0); del buf805  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf808, le_39, 65536, grid=grid(65536), stream=stream0)
        del le_39
        buf809 = reinterpret_tensor(buf35, (128, 128), (128, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf808, (128, 512), (512, 1), 0), permute_1135, out=buf809)
        del permute_1135
        buf797 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf802 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf803 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf807 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf812 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf813 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_220, attention_output_71, mul_117], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf794, buf799, addmm_220, buf34, primals_233, primals_234, buf804, buf809, buf797, buf802, buf803, buf807, buf812, buf813, 128, 128, grid=grid(128), stream=stream0)
        del addmm_220
        del buf34
        del primals_234
        buf800 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf798, (512, 128), (1, 512), 0), view_592, out=buf800)
        del view_592
        buf801 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf798, buf801, 512, 128, grid=grid(512), stream=stream0)
        buf806 = reinterpret_tensor(buf798, (128, 512), (512, 1), 0); del buf798  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf804, (128, 128), (1, 128), 0), view_590, out=buf806)
        del view_590
        buf810 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf808, (512, 128), (1, 512), 0), view_588, out=buf810)
        del view_588
        buf811 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf808, buf811, 512, 128, grid=grid(512), stream=stream0)
        buf814 = buf804; del buf804  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf814, buf809, primals_233, 16384, grid=grid(16384), stream=stream0)
        del primals_233
        buf815 = reinterpret_tensor(buf808, (128, 512), (512, 1), 0); del buf808  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf814, (128, 128), (128, 1), 0), permute_1139, out=buf815)
        del permute_1139
        buf816 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf814, (128, 128), (1, 128), 0), view_586, out=buf816)
        del view_586
        buf818 = reinterpret_tensor(buf815, (1, 128, 512), (65536, 512, 1), 0); del buf815  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf818, le_40, 65536, grid=grid(65536), stream=stream0)
        del le_40
        buf819 = buf809; del buf809  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf818, (128, 512), (512, 1), 0), permute_1143, out=buf819)
        del permute_1143
        buf824 = reinterpret_tensor(buf799, (1, 128, 128), (16384, 128, 1), 0); del buf799  # reuse
        buf849 = buf794; del buf794  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf814, buf819, primals_231, primals_227, buf824, buf849, 16384, grid=grid(16384), stream=stream0)
        del primals_231
        buf817 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf822 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf823 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf827 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf847 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf848 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_216, layer_input_74, mul_114], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf814, buf819, addmm_216, addmm_211, primals_227, primals_228, buf824, buf817, buf822, buf823, buf827, buf847, buf848, 128, 128, grid=grid(128), stream=stream0)
        del addmm_211
        del addmm_216
        del primals_227
        del primals_228
        buf820 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf818, (512, 128), (1, 512), 0), view_584, out=buf820)
        del view_584
        buf821 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf818, buf821, 512, 128, grid=grid(512), stream=stream0)
        buf825 = buf819; del buf819  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf824, (128, 128), (128, 1), 0), permute_1147, out=buf825)
        del permute_1147
        buf826 = reinterpret_tensor(buf814, (128, 128), (128, 1), 0); del buf814  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf824, (128, 128), (1, 128), 0), view_582, out=buf826)
        del view_582
        # Source Nodes: [], Original ATen: []
        buf828 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf825, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_27, clone_default_28, clone_default_29, None, alias_default_19, getitem_114, getitem_115, getitem_116, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_19
        del clone_default_27
        del clone_default_28
        del clone_default_29
        del getitem_114
        del getitem_115
        del getitem_116
        buf829 = buf828[0]
        buf830 = buf828[1]
        buf831 = buf828[2]
        del buf828
        buf832 = reinterpret_tensor(buf818, (128, 512), (512, 1), 0); del buf818  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf831, (128, 128), (128, 1), 0), permute_1160, out=buf832)
        del permute_1160
        buf833 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf831, (128, 128), (1, 128), 0), view_562, out=buf833)
        buf834 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf831, buf834, 128, 128, grid=grid(128), stream=stream0)
        buf835 = reinterpret_tensor(buf831, (128, 128), (128, 1), 0); del buf831  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf830, (128, 128), (128, 1), 0), permute_1164, out=buf835)
        del permute_1164
        buf836 = buf825; del buf825  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf830, (128, 128), (1, 128), 0), view_566, out=buf836)
        buf837 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf830, buf837, 128, 128, grid=grid(128), stream=stream0)
        buf838 = reinterpret_tensor(buf830, (128, 128), (128, 1), 0); del buf830  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf829, (128, 128), (128, 1), 0), permute_1168, out=buf838)
        del permute_1168
        buf839 = reinterpret_tensor(buf824, (128, 128), (128, 1), 0); del buf824  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf829, (128, 128), (1, 128), 0), view_566, out=buf839)
        del view_566
        buf840 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf829, buf840, 128, 128, grid=grid(128), stream=stream0)
        del buf829
        buf841 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf842 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf835, buf838, addmm_212, buf841, buf842, 128, 128, grid=grid(128), stream=stream0)
        del addmm_212
        buf843 = reinterpret_tensor(buf835, (1, 128, 128), (16384, 128, 1), 0); del buf835  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf843, buf838, primals_229, 16384, grid=grid(16384), stream=stream0)
        del primals_229
        buf844 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf843, (128, 128), (128, 1), 0), permute_1172, out=buf844)
        del permute_1172
        buf845 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf843, (128, 128), (1, 128), 0), view_562, out=buf845)
        buf846 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf843, buf846, 128, 128, grid=grid(128), stream=stream0)
        buf850 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf849, (128, 128), (128, 1), 0), permute_1176, out=buf850)
        del permute_1176
        buf851 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf849, (128, 128), (1, 128), 0), view_562, out=buf851)
        del view_562
        buf852 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf849, buf852, 128, 128, grid=grid(128), stream=stream0)
        buf853 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf855 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_211, mul_105, value_tensor_13], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_28.run(buf777, buf832, buf844, buf850, addmm_210, buf31, primals_209, primals_210, buf853, buf855, 512, 128, grid=grid(512), stream=stream0)
        del addmm_210
        del primals_210
        buf854 = buf777; del buf777  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf854, buf832, buf844, buf850, primals_225, 65536, grid=grid(65536), stream=stream0)
        del primals_225
        buf856 = reinterpret_tensor(buf849, (128, 128), (128, 1), 0); del buf849  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf854, (128, 512), (512, 1), 0), permute_1180, out=buf856)
        del permute_1180
        buf857 = reinterpret_tensor(buf850, (512, 128), (128, 1), 0); del buf850  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf854, (512, 128), (1, 512), 0), view_560, out=buf857)
        del view_560
        buf858 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf854, buf858, 512, 128, grid=grid(512), stream=stream0)
        buf861 = buf843; del buf843  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf856, primals_223, buf861, 16384, grid=grid(16384), stream=stream0)
        del primals_223
        buf862 = buf844; del buf844  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf861, (128, 128), (128, 1), 0), permute_1184, out=buf862)
        del permute_1184
        buf865 = reinterpret_tensor(buf862, (1, 128, 512), (65536, 512, 1), 0); del buf862  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf865, le_41, 65536, grid=grid(65536), stream=stream0)
        del le_41
        buf866 = buf838; del buf838  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf865, (128, 512), (512, 1), 0), permute_1188, out=buf866)
        del permute_1188
        buf859 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf860 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf864 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf869 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf870 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_209, attention_output_68, mul_111], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf856, addmm_209, buf33, primals_221, primals_222, buf861, buf866, buf859, buf860, buf864, buf869, buf870, 128, 128, grid=grid(128), stream=stream0)
        del addmm_209
        del primals_222
        buf863 = buf832; del buf832  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf861, (128, 128), (1, 128), 0), view_558, out=buf863)
        del view_558
        buf867 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf865, (512, 128), (1, 512), 0), view_556, out=buf867)
        del view_556
        buf868 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf865, buf868, 512, 128, grid=grid(512), stream=stream0)
        buf871 = buf861; del buf861  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf871, buf866, primals_221, 16384, grid=grid(16384), stream=stream0)
        del primals_221
        buf872 = reinterpret_tensor(buf865, (128, 512), (512, 1), 0); del buf865  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf871, (128, 128), (128, 1), 0), permute_1192, out=buf872)
        del permute_1192
        buf873 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf871, (128, 128), (1, 128), 0), view_554, out=buf873)
        del view_554
        buf875 = reinterpret_tensor(buf872, (1, 128, 512), (65536, 512, 1), 0); del buf872  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf875, le_42, 65536, grid=grid(65536), stream=stream0)
        del le_42
        buf876 = buf866; del buf866  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf875, (128, 512), (512, 1), 0), permute_1196, out=buf876)
        del permute_1196
        buf881 = reinterpret_tensor(buf856, (1, 128, 128), (16384, 128, 1), 0); del buf856  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf871, buf876, primals_219, buf881, 16384, grid=grid(16384), stream=stream0)
        del primals_219
        buf882 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf881, (128, 128), (128, 1), 0), permute_1200, out=buf882)
        del permute_1200
        buf885 = reinterpret_tensor(buf882, (1, 128, 512), (65536, 512, 1), 0); del buf882  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf885, le_43, 65536, grid=grid(65536), stream=stream0)
        del le_43
        buf886 = reinterpret_tensor(buf33, (128, 128), (128, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf885, (128, 512), (512, 1), 0), permute_1204, out=buf886)
        del permute_1204
        buf874 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf879 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf880 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf884 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf889 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf890 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_205, attention_output_66, mul_109], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf871, buf876, addmm_205, buf32, primals_217, primals_218, buf881, buf886, buf874, buf879, buf880, buf884, buf889, buf890, 128, 128, grid=grid(128), stream=stream0)
        del addmm_205
        del buf32
        del primals_218
        buf877 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf875, (512, 128), (1, 512), 0), view_552, out=buf877)
        del view_552
        buf878 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf875, buf878, 512, 128, grid=grid(512), stream=stream0)
        buf883 = reinterpret_tensor(buf875, (128, 512), (512, 1), 0); del buf875  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf881, (128, 128), (1, 128), 0), view_550, out=buf883)
        del view_550
        buf887 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf885, (512, 128), (1, 512), 0), view_548, out=buf887)
        del view_548
        buf888 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf885, buf888, 512, 128, grid=grid(512), stream=stream0)
        buf891 = buf881; del buf881  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf891, buf886, primals_217, 16384, grid=grid(16384), stream=stream0)
        del primals_217
        buf892 = reinterpret_tensor(buf885, (128, 512), (512, 1), 0); del buf885  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf891, (128, 128), (128, 1), 0), permute_1208, out=buf892)
        del permute_1208
        buf893 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf891, (128, 128), (1, 128), 0), view_546, out=buf893)
        del view_546
        buf895 = reinterpret_tensor(buf892, (1, 128, 512), (65536, 512, 1), 0); del buf892  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf895, le_44, 65536, grid=grid(65536), stream=stream0)
        del le_44
        buf896 = buf886; del buf886  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf895, (128, 512), (512, 1), 0), permute_1212, out=buf896)
        del permute_1212
        buf901 = reinterpret_tensor(buf876, (1, 128, 128), (16384, 128, 1), 0); del buf876  # reuse
        buf926 = buf871; del buf871  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf891, buf896, primals_215, primals_211, buf901, buf926, 16384, grid=grid(16384), stream=stream0)
        del primals_215
        buf894 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf899 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf900 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf904 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf924 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf925 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_201, layer_input_69, mul_106], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf891, buf896, addmm_201, addmm_196, primals_211, primals_212, buf901, buf894, buf899, buf900, buf904, buf924, buf925, 128, 128, grid=grid(128), stream=stream0)
        del addmm_196
        del addmm_201
        del primals_211
        del primals_212
        buf897 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf895, (512, 128), (1, 512), 0), view_544, out=buf897)
        del view_544
        buf898 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf895, buf898, 512, 128, grid=grid(512), stream=stream0)
        buf902 = buf896; del buf896  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf901, (128, 128), (128, 1), 0), permute_1216, out=buf902)
        del permute_1216
        buf903 = reinterpret_tensor(buf891, (128, 128), (128, 1), 0); del buf891  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf901, (128, 128), (1, 128), 0), view_542, out=buf903)
        del view_542
        # Source Nodes: [], Original ATen: []
        buf905 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf902, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_30, clone_default_31, clone_default_32, None, alias_default_21, getitem_121, getitem_122, getitem_123, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_21
        del clone_default_30
        del clone_default_31
        del clone_default_32
        del getitem_121
        del getitem_122
        del getitem_123
        buf906 = buf905[0]
        buf907 = buf905[1]
        buf908 = buf905[2]
        del buf905
        buf909 = reinterpret_tensor(buf895, (128, 512), (512, 1), 0); del buf895  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf908, (128, 128), (128, 1), 0), permute_1229, out=buf909)
        del permute_1229
        buf910 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf908, (128, 128), (1, 128), 0), view_522, out=buf910)
        buf911 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf908, buf911, 128, 128, grid=grid(128), stream=stream0)
        buf912 = reinterpret_tensor(buf908, (128, 128), (128, 1), 0); del buf908  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf907, (128, 128), (128, 1), 0), permute_1233, out=buf912)
        del permute_1233
        buf913 = buf902; del buf902  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf907, (128, 128), (1, 128), 0), view_526, out=buf913)
        buf914 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf907, buf914, 128, 128, grid=grid(128), stream=stream0)
        buf915 = reinterpret_tensor(buf907, (128, 128), (128, 1), 0); del buf907  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf906, (128, 128), (128, 1), 0), permute_1237, out=buf915)
        del permute_1237
        buf916 = reinterpret_tensor(buf901, (128, 128), (128, 1), 0); del buf901  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf906, (128, 128), (1, 128), 0), view_526, out=buf916)
        del view_526
        buf917 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf906, buf917, 128, 128, grid=grid(128), stream=stream0)
        del buf906
        buf918 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf919 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf912, buf915, addmm_197, buf918, buf919, 128, 128, grid=grid(128), stream=stream0)
        del addmm_197
        buf920 = reinterpret_tensor(buf912, (1, 128, 128), (16384, 128, 1), 0); del buf912  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf920, buf915, primals_213, 16384, grid=grid(16384), stream=stream0)
        del primals_213
        buf921 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf920, (128, 128), (128, 1), 0), permute_1241, out=buf921)
        del permute_1241
        buf922 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf920, (128, 128), (1, 128), 0), view_522, out=buf922)
        buf923 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf920, buf923, 128, 128, grid=grid(128), stream=stream0)
        buf927 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf926, (128, 128), (128, 1), 0), permute_1245, out=buf927)
        del permute_1245
        buf928 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf926, (128, 128), (1, 128), 0), view_522, out=buf928)
        del view_522
        buf929 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf926, buf929, 128, 128, grid=grid(128), stream=stream0)
        buf930 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf932 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_26.run(buf854, buf909, buf921, buf927, buf31, buf930, buf932, 512, 128, grid=grid(512), stream=stream0)
        buf931 = buf854; del buf854  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf931, buf909, buf921, buf927, primals_209, 65536, grid=grid(65536), stream=stream0)
        del primals_209
        buf933 = reinterpret_tensor(buf926, (128, 128), (128, 1), 0); del buf926  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf931, (128, 512), (512, 1), 0), permute_1249, out=buf933)
        del permute_1249
        buf934 = reinterpret_tensor(buf927, (512, 128), (128, 1), 0); del buf927  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf931, (512, 128), (1, 512), 0), view_520, out=buf934)
        del view_520
        buf935 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf931, buf935, 512, 128, grid=grid(512), stream=stream0)
        buf938 = buf920; del buf920  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf933, primals_207, buf938, 16384, grid=grid(16384), stream=stream0)
        del primals_207
        buf939 = buf921; del buf921  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf938, (128, 128), (128, 1), 0), permute_1253, out=buf939)
        del permute_1253
        buf942 = reinterpret_tensor(buf939, (1, 128, 512), (65536, 512, 1), 0); del buf939  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf942, le_45, 65536, grid=grid(65536), stream=stream0)
        del le_45
        buf943 = buf915; del buf915  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf942, (128, 512), (512, 1), 0), permute_1257, out=buf943)
        del permute_1257
        buf936 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf937 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf941 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf946 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf947 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_194, attention_output_63, mul_103], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf933, addmm_194, buf30, primals_205, primals_206, buf938, buf943, buf936, buf937, buf941, buf946, buf947, 128, 128, grid=grid(128), stream=stream0)
        del addmm_194
        del primals_206
        buf940 = buf909; del buf909  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf938, (128, 128), (1, 128), 0), view_518, out=buf940)
        del view_518
        buf944 = reinterpret_tensor(buf31, (512, 128), (128, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf942, (512, 128), (1, 512), 0), view_516, out=buf944)
        del view_516
        buf945 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf942, buf945, 512, 128, grid=grid(512), stream=stream0)
        buf948 = buf938; del buf938  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf948, buf943, primals_205, 16384, grid=grid(16384), stream=stream0)
        del primals_205
        buf949 = reinterpret_tensor(buf942, (128, 512), (512, 1), 0); del buf942  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf948, (128, 128), (128, 1), 0), permute_1261, out=buf949)
        del permute_1261
        buf950 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf948, (128, 128), (1, 128), 0), view_514, out=buf950)
        del view_514
        buf952 = reinterpret_tensor(buf949, (1, 128, 512), (65536, 512, 1), 0); del buf949  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf952, le_46, 65536, grid=grid(65536), stream=stream0)
        del le_46
        buf953 = buf943; del buf943  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf952, (128, 512), (512, 1), 0), permute_1265, out=buf953)
        del permute_1265
        buf958 = reinterpret_tensor(buf933, (1, 128, 128), (16384, 128, 1), 0); del buf933  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf948, buf953, primals_203, buf958, 16384, grid=grid(16384), stream=stream0)
        del primals_203
        buf959 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf958, (128, 128), (128, 1), 0), permute_1269, out=buf959)
        del permute_1269
        buf962 = reinterpret_tensor(buf959, (1, 128, 512), (65536, 512, 1), 0); del buf959  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf962, le_47, 65536, grid=grid(65536), stream=stream0)
        del le_47
        buf963 = reinterpret_tensor(buf30, (128, 128), (128, 1), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf962, (128, 512), (512, 1), 0), permute_1273, out=buf963)
        del permute_1273
        buf951 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf956 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf957 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf961 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf966 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf967 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_190, attention_output_61, mul_101], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf948, buf953, addmm_190, buf29, primals_201, primals_202, buf958, buf963, buf951, buf956, buf957, buf961, buf966, buf967, 128, 128, grid=grid(128), stream=stream0)
        del addmm_190
        del buf29
        del primals_202
        buf954 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf952, (512, 128), (1, 512), 0), view_512, out=buf954)
        del view_512
        buf955 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf952, buf955, 512, 128, grid=grid(512), stream=stream0)
        buf960 = reinterpret_tensor(buf952, (128, 512), (512, 1), 0); del buf952  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf958, (128, 128), (1, 128), 0), view_510, out=buf960)
        del view_510
        buf964 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf962, (512, 128), (1, 512), 0), view_508, out=buf964)
        del view_508
        buf965 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf962, buf965, 512, 128, grid=grid(512), stream=stream0)
        buf968 = buf958; del buf958  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf968, buf963, primals_201, 16384, grid=grid(16384), stream=stream0)
        del primals_201
        buf969 = reinterpret_tensor(buf962, (128, 512), (512, 1), 0); del buf962  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf968, (128, 128), (128, 1), 0), permute_1277, out=buf969)
        del permute_1277
        buf970 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf968, (128, 128), (1, 128), 0), view_506, out=buf970)
        del view_506
        buf972 = reinterpret_tensor(buf969, (1, 128, 512), (65536, 512, 1), 0); del buf969  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf972, le_48, 65536, grid=grid(65536), stream=stream0)
        del le_48
        buf973 = buf963; del buf963  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf972, (128, 512), (512, 1), 0), permute_1281, out=buf973)
        del permute_1281
        buf978 = reinterpret_tensor(buf953, (1, 128, 128), (16384, 128, 1), 0); del buf953  # reuse
        buf1003 = buf948; del buf948  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf968, buf973, primals_199, primals_195, buf978, buf1003, 16384, grid=grid(16384), stream=stream0)
        del primals_199
        buf971 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf976 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf977 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf981 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1001 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1002 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_186, layer_input_64, mul_98], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf968, buf973, addmm_186, addmm_181, primals_195, primals_196, buf978, buf971, buf976, buf977, buf981, buf1001, buf1002, 128, 128, grid=grid(128), stream=stream0)
        del addmm_181
        del addmm_186
        del primals_195
        del primals_196
        buf974 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf972, (512, 128), (1, 512), 0), view_504, out=buf974)
        del view_504
        buf975 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf972, buf975, 512, 128, grid=grid(512), stream=stream0)
        buf979 = buf973; del buf973  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf978, (128, 128), (128, 1), 0), permute_1285, out=buf979)
        del permute_1285
        buf980 = reinterpret_tensor(buf968, (128, 128), (128, 1), 0); del buf968  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf978, (128, 128), (1, 128), 0), view_502, out=buf980)
        del view_502
        # Source Nodes: [], Original ATen: []
        buf982 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf979, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_33, clone_default_34, clone_default_35, None, alias_default_23, getitem_128, getitem_129, getitem_130, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_23
        del clone_default_33
        del clone_default_34
        del clone_default_35
        del getitem_128
        del getitem_129
        del getitem_130
        buf983 = buf982[0]
        buf984 = buf982[1]
        buf985 = buf982[2]
        del buf982
        buf986 = reinterpret_tensor(buf972, (128, 512), (512, 1), 0); del buf972  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf985, (128, 128), (128, 1), 0), permute_1298, out=buf986)
        del permute_1298
        buf987 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf985, (128, 128), (1, 128), 0), view_482, out=buf987)
        buf988 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf985, buf988, 128, 128, grid=grid(128), stream=stream0)
        buf989 = reinterpret_tensor(buf985, (128, 128), (128, 1), 0); del buf985  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf984, (128, 128), (128, 1), 0), permute_1302, out=buf989)
        del permute_1302
        buf990 = buf979; del buf979  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf984, (128, 128), (1, 128), 0), view_486, out=buf990)
        buf991 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf984, buf991, 128, 128, grid=grid(128), stream=stream0)
        buf992 = reinterpret_tensor(buf984, (128, 128), (128, 1), 0); del buf984  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf983, (128, 128), (128, 1), 0), permute_1306, out=buf992)
        del permute_1306
        buf993 = reinterpret_tensor(buf978, (128, 128), (128, 1), 0); del buf978  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf983, (128, 128), (1, 128), 0), view_486, out=buf993)
        del view_486
        buf994 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf983, buf994, 128, 128, grid=grid(128), stream=stream0)
        del buf983
        buf995 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf996 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf989, buf992, addmm_182, buf995, buf996, 128, 128, grid=grid(128), stream=stream0)
        del addmm_182
        buf997 = reinterpret_tensor(buf989, (1, 128, 128), (16384, 128, 1), 0); del buf989  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf997, buf992, primals_197, 16384, grid=grid(16384), stream=stream0)
        del primals_197
        buf998 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf997, (128, 128), (128, 1), 0), permute_1310, out=buf998)
        del permute_1310
        buf999 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf997, (128, 128), (1, 128), 0), view_482, out=buf999)
        buf1000 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf997, buf1000, 128, 128, grid=grid(128), stream=stream0)
        buf1004 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1003, (128, 128), (128, 1), 0), permute_1314, out=buf1004)
        del permute_1314
        buf1005 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1003, (128, 128), (1, 128), 0), view_482, out=buf1005)
        del view_482
        buf1006 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1003, buf1006, 128, 128, grid=grid(128), stream=stream0)
        buf1007 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1009 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_181, mul_89, value_tensor_11], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_28.run(buf931, buf986, buf998, buf1004, addmm_180, buf26, primals_177, primals_178, buf1007, buf1009, 512, 128, grid=grid(512), stream=stream0)
        del addmm_180
        del primals_178
        buf1008 = reinterpret_tensor(buf1004, (1, 128, 512), (65536, 512, 1), 0); del buf1004  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_30.run(buf1008, buf931, buf986, buf998, primals_193, 65536, grid=grid(65536), stream=stream0)
        del primals_193
        buf1010 = reinterpret_tensor(buf1003, (128, 128), (128, 1), 0); del buf1003  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1008, (128, 512), (512, 1), 0), permute_1318, out=buf1010)
        del permute_1318
        buf1011 = reinterpret_tensor(buf998, (512, 128), (128, 1), 0); del buf998  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1008, (512, 128), (1, 512), 0), view_480, out=buf1011)
        del view_480
        buf1012 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf1008, buf1012, 512, 128, grid=grid(512), stream=stream0)
        buf1015 = buf997; del buf997  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf1010, primals_191, buf1015, 16384, grid=grid(16384), stream=stream0)
        del primals_191
        buf1016 = buf986; del buf986  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1015, (128, 128), (128, 1), 0), permute_1322, out=buf1016)
        del permute_1322
        buf1019 = reinterpret_tensor(buf1016, (1, 128, 512), (65536, 512, 1), 0); del buf1016  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1019, le_49, 65536, grid=grid(65536), stream=stream0)
        del le_49
        buf1020 = buf992; del buf992  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1019, (128, 512), (512, 1), 0), permute_1326, out=buf1020)
        del permute_1326
        buf1013 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1014 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1018 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1023 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1024 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_179, attention_output_58, mul_95], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf1010, addmm_179, buf28, primals_189, primals_190, buf1015, buf1020, buf1013, buf1014, buf1018, buf1023, buf1024, 128, 128, grid=grid(128), stream=stream0)
        del addmm_179
        del primals_190
        buf1017 = reinterpret_tensor(buf931, (128, 512), (512, 1), 0); del buf931  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1015, (128, 128), (1, 128), 0), view_478, out=buf1017)
        del view_478
        buf1021 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1019, (512, 128), (1, 512), 0), view_476, out=buf1021)
        del view_476
        buf1022 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1019, buf1022, 512, 128, grid=grid(512), stream=stream0)
        buf1025 = buf1015; del buf1015  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1025, buf1020, primals_189, 16384, grid=grid(16384), stream=stream0)
        del primals_189
        buf1026 = reinterpret_tensor(buf1019, (128, 512), (512, 1), 0); del buf1019  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1025, (128, 128), (128, 1), 0), permute_1330, out=buf1026)
        del permute_1330
        buf1027 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1025, (128, 128), (1, 128), 0), view_474, out=buf1027)
        del view_474
        buf1029 = reinterpret_tensor(buf1026, (1, 128, 512), (65536, 512, 1), 0); del buf1026  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1029, le_50, 65536, grid=grid(65536), stream=stream0)
        del le_50
        buf1030 = buf1020; del buf1020  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1029, (128, 512), (512, 1), 0), permute_1334, out=buf1030)
        del permute_1334
        buf1035 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf1025, buf1030, primals_187, buf1035, 16384, grid=grid(16384), stream=stream0)
        del primals_187
        buf1036 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1035, (128, 128), (128, 1), 0), permute_1338, out=buf1036)
        del permute_1338
        buf1039 = reinterpret_tensor(buf1036, (1, 128, 512), (65536, 512, 1), 0); del buf1036  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1039, le_51, 65536, grid=grid(65536), stream=stream0)
        del le_51
        buf1040 = buf1010; del buf1010  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1039, (128, 512), (512, 1), 0), permute_1342, out=buf1040)
        del permute_1342
        buf1028 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1033 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1034 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1038 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1043 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1044 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_175, attention_output_56, mul_93], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf1025, buf1030, addmm_175, buf27, primals_185, primals_186, buf1035, buf1040, buf1028, buf1033, buf1034, buf1038, buf1043, buf1044, 128, 128, grid=grid(128), stream=stream0)
        del addmm_175
        del buf1025
        del primals_186
        buf1031 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1029, (512, 128), (1, 512), 0), view_472, out=buf1031)
        del view_472
        buf1032 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1029, buf1032, 512, 128, grid=grid(512), stream=stream0)
        buf1037 = reinterpret_tensor(buf1029, (128, 512), (512, 1), 0); del buf1029  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1035, (128, 128), (1, 128), 0), view_470, out=buf1037)
        del view_470
        buf1041 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1039, (512, 128), (1, 512), 0), view_468, out=buf1041)
        del view_468
        buf1042 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1039, buf1042, 512, 128, grid=grid(512), stream=stream0)
        buf1045 = buf1035; del buf1035  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1045, buf1040, primals_185, 16384, grid=grid(16384), stream=stream0)
        del primals_185
        buf1046 = reinterpret_tensor(buf1039, (128, 512), (512, 1), 0); del buf1039  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1045, (128, 128), (128, 1), 0), permute_1346, out=buf1046)
        del permute_1346
        buf1047 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1045, (128, 128), (1, 128), 0), view_466, out=buf1047)
        del view_466
        buf1049 = reinterpret_tensor(buf1046, (1, 128, 512), (65536, 512, 1), 0); del buf1046  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1049, le_52, 65536, grid=grid(65536), stream=stream0)
        del le_52
        buf1050 = buf1040; del buf1040  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1049, (128, 512), (512, 1), 0), permute_1350, out=buf1050)
        del permute_1350
        buf1055 = buf27; del buf27  # reuse
        buf1080 = reinterpret_tensor(buf1030, (1, 128, 128), (16384, 128, 1), 0); del buf1030  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf1045, buf1050, primals_183, primals_179, buf1055, buf1080, 16384, grid=grid(16384), stream=stream0)
        del primals_183
        buf1048 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1053 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1054 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1058 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1078 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1079 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_171, layer_input_59, mul_90], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf1045, buf1050, addmm_171, addmm_166, primals_179, primals_180, buf1055, buf1048, buf1053, buf1054, buf1058, buf1078, buf1079, 128, 128, grid=grid(128), stream=stream0)
        del addmm_166
        del addmm_171
        del primals_179
        del primals_180
        buf1051 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1049, (512, 128), (1, 512), 0), view_464, out=buf1051)
        del view_464
        buf1052 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1049, buf1052, 512, 128, grid=grid(512), stream=stream0)
        buf1056 = buf1050; del buf1050  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1055, (128, 128), (128, 1), 0), permute_1354, out=buf1056)
        del permute_1354
        buf1057 = reinterpret_tensor(buf1045, (128, 128), (128, 1), 0); del buf1045  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1055, (128, 128), (1, 128), 0), view_462, out=buf1057)
        del view_462
        # Source Nodes: [], Original ATen: []
        buf1059 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1056, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_36, clone_default_37, clone_default_38, None, alias_default_25, getitem_135, getitem_136, getitem_137, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_25
        del clone_default_36
        del clone_default_37
        del clone_default_38
        del getitem_135
        del getitem_136
        del getitem_137
        buf1060 = buf1059[0]
        buf1061 = buf1059[1]
        buf1062 = buf1059[2]
        del buf1059
        buf1063 = reinterpret_tensor(buf1049, (128, 512), (512, 1), 0); del buf1049  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1062, (128, 128), (128, 1), 0), permute_1367, out=buf1063)
        del permute_1367
        buf1064 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1062, (128, 128), (1, 128), 0), view_442, out=buf1064)
        buf1065 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1062, buf1065, 128, 128, grid=grid(128), stream=stream0)
        buf1066 = reinterpret_tensor(buf1062, (128, 128), (128, 1), 0); del buf1062  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1061, (128, 128), (128, 1), 0), permute_1371, out=buf1066)
        del permute_1371
        buf1067 = buf1056; del buf1056  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1061, (128, 128), (1, 128), 0), view_446, out=buf1067)
        buf1068 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1061, buf1068, 128, 128, grid=grid(128), stream=stream0)
        buf1069 = reinterpret_tensor(buf1061, (128, 128), (128, 1), 0); del buf1061  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1060, (128, 128), (128, 1), 0), permute_1375, out=buf1069)
        del permute_1375
        buf1070 = reinterpret_tensor(buf1055, (128, 128), (128, 1), 0); del buf1055  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1060, (128, 128), (1, 128), 0), view_446, out=buf1070)
        del view_446
        buf1071 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1060, buf1071, 128, 128, grid=grid(128), stream=stream0)
        del buf1060
        buf1072 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1073 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf1066, buf1069, addmm_167, buf1072, buf1073, 128, 128, grid=grid(128), stream=stream0)
        del addmm_167
        buf1074 = reinterpret_tensor(buf1066, (1, 128, 128), (16384, 128, 1), 0); del buf1066  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1074, buf1069, primals_181, 16384, grid=grid(16384), stream=stream0)
        del primals_181
        buf1075 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1074, (128, 128), (128, 1), 0), permute_1379, out=buf1075)
        del permute_1379
        buf1076 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1074, (128, 128), (1, 128), 0), view_442, out=buf1076)
        buf1077 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1074, buf1077, 128, 128, grid=grid(128), stream=stream0)
        buf1081 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1080, (128, 128), (128, 1), 0), permute_1383, out=buf1081)
        del permute_1383
        buf1082 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1080, (128, 128), (1, 128), 0), view_442, out=buf1082)
        del view_442
        buf1083 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1080, buf1083, 128, 128, grid=grid(128), stream=stream0)
        buf1084 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1086 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_26.run(buf1008, buf1063, buf1075, buf1081, buf26, buf1084, buf1086, 512, 128, grid=grid(512), stream=stream0)
        buf1085 = buf1008; del buf1008  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf1085, buf1063, buf1075, buf1081, primals_177, 65536, grid=grid(65536), stream=stream0)
        del primals_177
        buf1087 = reinterpret_tensor(buf1080, (128, 128), (128, 1), 0); del buf1080  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1085, (128, 512), (512, 1), 0), permute_1387, out=buf1087)
        del permute_1387
        buf1088 = reinterpret_tensor(buf1081, (512, 128), (128, 1), 0); del buf1081  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1085, (512, 128), (1, 512), 0), view_440, out=buf1088)
        del view_440
        buf1089 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf1085, buf1089, 512, 128, grid=grid(512), stream=stream0)
        buf1092 = buf1074; del buf1074  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf1087, primals_175, buf1092, 16384, grid=grid(16384), stream=stream0)
        del primals_175
        buf1093 = buf1075; del buf1075  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1092, (128, 128), (128, 1), 0), permute_1391, out=buf1093)
        del permute_1391
        buf1096 = reinterpret_tensor(buf1093, (1, 128, 512), (65536, 512, 1), 0); del buf1093  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1096, le_53, 65536, grid=grid(65536), stream=stream0)
        del le_53
        buf1097 = buf1069; del buf1069  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1096, (128, 512), (512, 1), 0), permute_1395, out=buf1097)
        del permute_1395
        buf1090 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1091 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1095 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1100 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1101 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_164, attention_output_53, mul_87], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf1087, addmm_164, buf25, primals_173, primals_174, buf1092, buf1097, buf1090, buf1091, buf1095, buf1100, buf1101, 128, 128, grid=grid(128), stream=stream0)
        del addmm_164
        del primals_174
        buf1094 = buf1063; del buf1063  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1092, (128, 128), (1, 128), 0), view_438, out=buf1094)
        del view_438
        buf1098 = reinterpret_tensor(buf26, (512, 128), (128, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1096, (512, 128), (1, 512), 0), view_436, out=buf1098)
        del view_436
        buf1099 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1096, buf1099, 512, 128, grid=grid(512), stream=stream0)
        buf1102 = buf1092; del buf1092  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1102, buf1097, primals_173, 16384, grid=grid(16384), stream=stream0)
        del primals_173
        buf1103 = reinterpret_tensor(buf1096, (128, 512), (512, 1), 0); del buf1096  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1102, (128, 128), (128, 1), 0), permute_1399, out=buf1103)
        del permute_1399
        buf1104 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1102, (128, 128), (1, 128), 0), view_434, out=buf1104)
        del view_434
        buf1106 = reinterpret_tensor(buf1103, (1, 128, 512), (65536, 512, 1), 0); del buf1103  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1106, le_54, 65536, grid=grid(65536), stream=stream0)
        del le_54
        buf1107 = buf1097; del buf1097  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1106, (128, 512), (512, 1), 0), permute_1403, out=buf1107)
        del permute_1403
        buf1112 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf1102, buf1107, primals_171, buf1112, 16384, grid=grid(16384), stream=stream0)
        del primals_171
        buf1113 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1112, (128, 128), (128, 1), 0), permute_1407, out=buf1113)
        del permute_1407
        buf1116 = reinterpret_tensor(buf1113, (1, 128, 512), (65536, 512, 1), 0); del buf1113  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1116, le_55, 65536, grid=grid(65536), stream=stream0)
        del le_55
        buf1117 = buf1087; del buf1087  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1116, (128, 512), (512, 1), 0), permute_1411, out=buf1117)
        del permute_1411
        buf1105 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1110 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1111 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1115 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1120 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1121 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_160, attention_output_51, mul_85], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf1102, buf1107, addmm_160, buf24, primals_169, primals_170, buf1112, buf1117, buf1105, buf1110, buf1111, buf1115, buf1120, buf1121, 128, 128, grid=grid(128), stream=stream0)
        del addmm_160
        del buf1102
        del primals_170
        buf1108 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1106, (512, 128), (1, 512), 0), view_432, out=buf1108)
        del view_432
        buf1109 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1106, buf1109, 512, 128, grid=grid(512), stream=stream0)
        buf1114 = reinterpret_tensor(buf1106, (128, 512), (512, 1), 0); del buf1106  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1112, (128, 128), (1, 128), 0), view_430, out=buf1114)
        del view_430
        buf1118 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1116, (512, 128), (1, 512), 0), view_428, out=buf1118)
        del view_428
        buf1119 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1116, buf1119, 512, 128, grid=grid(512), stream=stream0)
        buf1122 = buf1112; del buf1112  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1122, buf1117, primals_169, 16384, grid=grid(16384), stream=stream0)
        del primals_169
        buf1123 = reinterpret_tensor(buf1116, (128, 512), (512, 1), 0); del buf1116  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1122, (128, 128), (128, 1), 0), permute_1415, out=buf1123)
        del permute_1415
        buf1124 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1122, (128, 128), (1, 128), 0), view_426, out=buf1124)
        del view_426
        buf1126 = reinterpret_tensor(buf1123, (1, 128, 512), (65536, 512, 1), 0); del buf1123  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1126, le_56, 65536, grid=grid(65536), stream=stream0)
        del le_56
        buf1127 = buf1117; del buf1117  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1126, (128, 512), (512, 1), 0), permute_1419, out=buf1127)
        del permute_1419
        buf1132 = buf24; del buf24  # reuse
        buf1157 = reinterpret_tensor(buf1107, (1, 128, 128), (16384, 128, 1), 0); del buf1107  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf1122, buf1127, primals_167, primals_163, buf1132, buf1157, 16384, grid=grid(16384), stream=stream0)
        del primals_167
        buf1125 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1130 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1131 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1135 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1155 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1156 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_156, layer_input_54, mul_82], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf1122, buf1127, addmm_156, addmm_151, primals_163, primals_164, buf1132, buf1125, buf1130, buf1131, buf1135, buf1155, buf1156, 128, 128, grid=grid(128), stream=stream0)
        del addmm_151
        del addmm_156
        del primals_163
        del primals_164
        buf1128 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1126, (512, 128), (1, 512), 0), view_424, out=buf1128)
        del view_424
        buf1129 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1126, buf1129, 512, 128, grid=grid(512), stream=stream0)
        buf1133 = buf1127; del buf1127  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1132, (128, 128), (128, 1), 0), permute_1423, out=buf1133)
        del permute_1423
        buf1134 = reinterpret_tensor(buf1122, (128, 128), (128, 1), 0); del buf1122  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1132, (128, 128), (1, 128), 0), view_422, out=buf1134)
        del view_422
        # Source Nodes: [], Original ATen: []
        buf1136 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1133, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_39, clone_default_40, clone_default_41, None, alias_default_27, getitem_142, getitem_143, getitem_144, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_27
        del clone_default_39
        del clone_default_40
        del clone_default_41
        del getitem_142
        del getitem_143
        del getitem_144
        buf1137 = buf1136[0]
        buf1138 = buf1136[1]
        buf1139 = buf1136[2]
        del buf1136
        buf1140 = reinterpret_tensor(buf1126, (128, 512), (512, 1), 0); del buf1126  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1139, (128, 128), (128, 1), 0), permute_1436, out=buf1140)
        del permute_1436
        buf1141 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1139, (128, 128), (1, 128), 0), view_402, out=buf1141)
        buf1142 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1139, buf1142, 128, 128, grid=grid(128), stream=stream0)
        buf1143 = reinterpret_tensor(buf1139, (128, 128), (128, 1), 0); del buf1139  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1138, (128, 128), (128, 1), 0), permute_1440, out=buf1143)
        del permute_1440
        buf1144 = buf1133; del buf1133  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1138, (128, 128), (1, 128), 0), view_406, out=buf1144)
        buf1145 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1138, buf1145, 128, 128, grid=grid(128), stream=stream0)
        buf1146 = reinterpret_tensor(buf1138, (128, 128), (128, 1), 0); del buf1138  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1137, (128, 128), (128, 1), 0), permute_1444, out=buf1146)
        del permute_1444
        buf1147 = reinterpret_tensor(buf1132, (128, 128), (128, 1), 0); del buf1132  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1137, (128, 128), (1, 128), 0), view_406, out=buf1147)
        del view_406
        buf1148 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1137, buf1148, 128, 128, grid=grid(128), stream=stream0)
        del buf1137
        buf1149 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1150 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf1143, buf1146, addmm_152, buf1149, buf1150, 128, 128, grid=grid(128), stream=stream0)
        del addmm_152
        buf1151 = reinterpret_tensor(buf1143, (1, 128, 128), (16384, 128, 1), 0); del buf1143  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1151, buf1146, primals_165, 16384, grid=grid(16384), stream=stream0)
        del primals_165
        buf1152 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1151, (128, 128), (128, 1), 0), permute_1448, out=buf1152)
        del permute_1448
        buf1153 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1151, (128, 128), (1, 128), 0), view_402, out=buf1153)
        buf1154 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1151, buf1154, 128, 128, grid=grid(128), stream=stream0)
        buf1158 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1157, (128, 128), (128, 1), 0), permute_1452, out=buf1158)
        del permute_1452
        buf1159 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1157, (128, 128), (1, 128), 0), view_402, out=buf1159)
        del view_402
        buf1160 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1157, buf1160, 128, 128, grid=grid(128), stream=stream0)
        buf1161 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1163 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_151, mul_73, value_tensor_9], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_28.run(buf1085, buf1140, buf1152, buf1158, addmm_150, buf21, primals_145, primals_146, buf1161, buf1163, 512, 128, grid=grid(512), stream=stream0)
        del addmm_150
        del primals_146
        buf1162 = buf1085; del buf1085  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf1162, buf1140, buf1152, buf1158, primals_161, 65536, grid=grid(65536), stream=stream0)
        del primals_161
        buf1164 = reinterpret_tensor(buf1157, (128, 128), (128, 1), 0); del buf1157  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1162, (128, 512), (512, 1), 0), permute_1456, out=buf1164)
        del permute_1456
        buf1165 = reinterpret_tensor(buf1158, (512, 128), (128, 1), 0); del buf1158  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1162, (512, 128), (1, 512), 0), view_400, out=buf1165)
        del view_400
        buf1166 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf1162, buf1166, 512, 128, grid=grid(512), stream=stream0)
        buf1169 = buf1151; del buf1151  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf1164, primals_159, buf1169, 16384, grid=grid(16384), stream=stream0)
        del primals_159
        buf1170 = buf1152; del buf1152  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1169, (128, 128), (128, 1), 0), permute_1460, out=buf1170)
        del permute_1460
        buf1173 = reinterpret_tensor(buf1170, (1, 128, 512), (65536, 512, 1), 0); del buf1170  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1173, le_57, 65536, grid=grid(65536), stream=stream0)
        del le_57
        buf1174 = buf1146; del buf1146  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1173, (128, 512), (512, 1), 0), permute_1464, out=buf1174)
        del permute_1464
        buf1167 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1168 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1172 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1177 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1178 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_149, attention_output_48, mul_79], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf1164, addmm_149, buf23, primals_157, primals_158, buf1169, buf1174, buf1167, buf1168, buf1172, buf1177, buf1178, 128, 128, grid=grid(128), stream=stream0)
        del addmm_149
        del primals_158
        buf1171 = buf1140; del buf1140  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1169, (128, 128), (1, 128), 0), view_398, out=buf1171)
        del view_398
        buf1175 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1173, (512, 128), (1, 512), 0), view_396, out=buf1175)
        del view_396
        buf1176 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1173, buf1176, 512, 128, grid=grid(512), stream=stream0)
        buf1179 = buf1169; del buf1169  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1179, buf1174, primals_157, 16384, grid=grid(16384), stream=stream0)
        del primals_157
        buf1180 = reinterpret_tensor(buf1173, (128, 512), (512, 1), 0); del buf1173  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1179, (128, 128), (128, 1), 0), permute_1468, out=buf1180)
        del permute_1468
        buf1181 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1179, (128, 128), (1, 128), 0), view_394, out=buf1181)
        del view_394
        buf1183 = reinterpret_tensor(buf1180, (1, 128, 512), (65536, 512, 1), 0); del buf1180  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1183, le_58, 65536, grid=grid(65536), stream=stream0)
        del le_58
        buf1184 = buf1174; del buf1174  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1183, (128, 512), (512, 1), 0), permute_1472, out=buf1184)
        del permute_1472
        buf1189 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf1179, buf1184, primals_155, buf1189, 16384, grid=grid(16384), stream=stream0)
        del primals_155
        buf1190 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1189, (128, 128), (128, 1), 0), permute_1476, out=buf1190)
        del permute_1476
        buf1193 = reinterpret_tensor(buf1190, (1, 128, 512), (65536, 512, 1), 0); del buf1190  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1193, le_59, 65536, grid=grid(65536), stream=stream0)
        del le_59
        buf1194 = buf1164; del buf1164  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1193, (128, 512), (512, 1), 0), permute_1480, out=buf1194)
        del permute_1480
        buf1182 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1187 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1188 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1192 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1197 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1198 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_145, attention_output_46, mul_77], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf1179, buf1184, addmm_145, buf22, primals_153, primals_154, buf1189, buf1194, buf1182, buf1187, buf1188, buf1192, buf1197, buf1198, 128, 128, grid=grid(128), stream=stream0)
        del addmm_145
        del buf1179
        del primals_154
        buf1185 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1183, (512, 128), (1, 512), 0), view_392, out=buf1185)
        del view_392
        buf1186 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1183, buf1186, 512, 128, grid=grid(512), stream=stream0)
        buf1191 = reinterpret_tensor(buf1183, (128, 512), (512, 1), 0); del buf1183  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1189, (128, 128), (1, 128), 0), view_390, out=buf1191)
        del view_390
        buf1195 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1193, (512, 128), (1, 512), 0), view_388, out=buf1195)
        del view_388
        buf1196 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1193, buf1196, 512, 128, grid=grid(512), stream=stream0)
        buf1199 = buf1189; del buf1189  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1199, buf1194, primals_153, 16384, grid=grid(16384), stream=stream0)
        del primals_153
        buf1200 = reinterpret_tensor(buf1193, (128, 512), (512, 1), 0); del buf1193  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1199, (128, 128), (128, 1), 0), permute_1484, out=buf1200)
        del permute_1484
        buf1201 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1199, (128, 128), (1, 128), 0), view_386, out=buf1201)
        del view_386
        buf1203 = reinterpret_tensor(buf1200, (1, 128, 512), (65536, 512, 1), 0); del buf1200  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1203, le_60, 65536, grid=grid(65536), stream=stream0)
        del le_60
        buf1204 = buf1194; del buf1194  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1203, (128, 512), (512, 1), 0), permute_1488, out=buf1204)
        del permute_1488
        buf1209 = buf22; del buf22  # reuse
        buf1234 = reinterpret_tensor(buf1184, (1, 128, 128), (16384, 128, 1), 0); del buf1184  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf1199, buf1204, primals_151, primals_147, buf1209, buf1234, 16384, grid=grid(16384), stream=stream0)
        del primals_151
        buf1202 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1207 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1208 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1212 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1232 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1233 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_141, layer_input_49, mul_74], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf1199, buf1204, addmm_141, addmm_136, primals_147, primals_148, buf1209, buf1202, buf1207, buf1208, buf1212, buf1232, buf1233, 128, 128, grid=grid(128), stream=stream0)
        del addmm_136
        del addmm_141
        del primals_147
        del primals_148
        buf1205 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1203, (512, 128), (1, 512), 0), view_384, out=buf1205)
        del view_384
        buf1206 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1203, buf1206, 512, 128, grid=grid(512), stream=stream0)
        buf1210 = buf1204; del buf1204  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1209, (128, 128), (128, 1), 0), permute_1492, out=buf1210)
        del permute_1492
        buf1211 = reinterpret_tensor(buf1199, (128, 128), (128, 1), 0); del buf1199  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1209, (128, 128), (1, 128), 0), view_382, out=buf1211)
        del view_382
        # Source Nodes: [], Original ATen: []
        buf1213 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1210, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_42, clone_default_43, clone_default_44, None, alias_default_29, getitem_149, getitem_150, getitem_151, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_29
        del clone_default_42
        del clone_default_43
        del clone_default_44
        del getitem_149
        del getitem_150
        del getitem_151
        buf1214 = buf1213[0]
        buf1215 = buf1213[1]
        buf1216 = buf1213[2]
        del buf1213
        buf1217 = reinterpret_tensor(buf1203, (128, 512), (512, 1), 0); del buf1203  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1216, (128, 128), (128, 1), 0), permute_1505, out=buf1217)
        del permute_1505
        buf1218 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1216, (128, 128), (1, 128), 0), view_362, out=buf1218)
        buf1219 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1216, buf1219, 128, 128, grid=grid(128), stream=stream0)
        buf1220 = reinterpret_tensor(buf1216, (128, 128), (128, 1), 0); del buf1216  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1215, (128, 128), (128, 1), 0), permute_1509, out=buf1220)
        del permute_1509
        buf1221 = buf1210; del buf1210  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1215, (128, 128), (1, 128), 0), view_366, out=buf1221)
        buf1222 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1215, buf1222, 128, 128, grid=grid(128), stream=stream0)
        buf1223 = reinterpret_tensor(buf1215, (128, 128), (128, 1), 0); del buf1215  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1214, (128, 128), (128, 1), 0), permute_1513, out=buf1223)
        del permute_1513
        buf1224 = reinterpret_tensor(buf1209, (128, 128), (128, 1), 0); del buf1209  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1214, (128, 128), (1, 128), 0), view_366, out=buf1224)
        del view_366
        buf1225 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1214, buf1225, 128, 128, grid=grid(128), stream=stream0)
        del buf1214
        buf1226 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1227 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf1220, buf1223, addmm_137, buf1226, buf1227, 128, 128, grid=grid(128), stream=stream0)
        del addmm_137
        buf1228 = reinterpret_tensor(buf1220, (1, 128, 128), (16384, 128, 1), 0); del buf1220  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1228, buf1223, primals_149, 16384, grid=grid(16384), stream=stream0)
        del primals_149
        buf1229 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1228, (128, 128), (128, 1), 0), permute_1517, out=buf1229)
        del permute_1517
        buf1230 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1228, (128, 128), (1, 128), 0), view_362, out=buf1230)
        buf1231 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1228, buf1231, 128, 128, grid=grid(128), stream=stream0)
        buf1235 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1234, (128, 128), (128, 1), 0), permute_1521, out=buf1235)
        del permute_1521
        buf1236 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1234, (128, 128), (1, 128), 0), view_362, out=buf1236)
        del view_362
        buf1237 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1234, buf1237, 128, 128, grid=grid(128), stream=stream0)
        buf1238 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1240 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_26.run(buf1162, buf1217, buf1229, buf1235, buf21, buf1238, buf1240, 512, 128, grid=grid(512), stream=stream0)
        buf1239 = buf1162; del buf1162  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf1239, buf1217, buf1229, buf1235, primals_145, 65536, grid=grid(65536), stream=stream0)
        del primals_145
        buf1241 = reinterpret_tensor(buf1234, (128, 128), (128, 1), 0); del buf1234  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1239, (128, 512), (512, 1), 0), permute_1525, out=buf1241)
        del permute_1525
        buf1242 = reinterpret_tensor(buf1235, (512, 128), (128, 1), 0); del buf1235  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1239, (512, 128), (1, 512), 0), view_360, out=buf1242)
        del view_360
        buf1243 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf1239, buf1243, 512, 128, grid=grid(512), stream=stream0)
        buf1246 = buf1228; del buf1228  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf1241, primals_143, buf1246, 16384, grid=grid(16384), stream=stream0)
        del primals_143
        buf1247 = buf1229; del buf1229  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1246, (128, 128), (128, 1), 0), permute_1529, out=buf1247)
        del permute_1529
        buf1250 = reinterpret_tensor(buf1247, (1, 128, 512), (65536, 512, 1), 0); del buf1247  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1250, le_61, 65536, grid=grid(65536), stream=stream0)
        del le_61
        buf1251 = buf1223; del buf1223  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1250, (128, 512), (512, 1), 0), permute_1533, out=buf1251)
        del permute_1533
        buf1244 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1245 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1249 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1254 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1255 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_134, attention_output_43, mul_71], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf1241, addmm_134, buf20, primals_141, primals_142, buf1246, buf1251, buf1244, buf1245, buf1249, buf1254, buf1255, 128, 128, grid=grid(128), stream=stream0)
        del addmm_134
        del primals_142
        buf1248 = buf1217; del buf1217  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1246, (128, 128), (1, 128), 0), view_358, out=buf1248)
        del view_358
        buf1252 = reinterpret_tensor(buf21, (512, 128), (128, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1250, (512, 128), (1, 512), 0), view_356, out=buf1252)
        del view_356
        buf1253 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1250, buf1253, 512, 128, grid=grid(512), stream=stream0)
        buf1256 = buf1246; del buf1246  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1256, buf1251, primals_141, 16384, grid=grid(16384), stream=stream0)
        del primals_141
        buf1257 = reinterpret_tensor(buf1250, (128, 512), (512, 1), 0); del buf1250  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1256, (128, 128), (128, 1), 0), permute_1537, out=buf1257)
        del permute_1537
        buf1258 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1256, (128, 128), (1, 128), 0), view_354, out=buf1258)
        del view_354
        buf1260 = reinterpret_tensor(buf1257, (1, 128, 512), (65536, 512, 1), 0); del buf1257  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1260, le_62, 65536, grid=grid(65536), stream=stream0)
        del le_62
        buf1261 = buf1251; del buf1251  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1260, (128, 512), (512, 1), 0), permute_1541, out=buf1261)
        del permute_1541
        buf1266 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf1256, buf1261, primals_139, buf1266, 16384, grid=grid(16384), stream=stream0)
        del primals_139
        buf1267 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1266, (128, 128), (128, 1), 0), permute_1545, out=buf1267)
        del permute_1545
        buf1270 = reinterpret_tensor(buf1267, (1, 128, 512), (65536, 512, 1), 0); del buf1267  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1270, le_63, 65536, grid=grid(65536), stream=stream0)
        del le_63
        buf1271 = buf1241; del buf1241  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1270, (128, 512), (512, 1), 0), permute_1549, out=buf1271)
        del permute_1549
        buf1259 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1264 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1265 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1269 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1274 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1275 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_130, attention_output_41, mul_69], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf1256, buf1261, addmm_130, buf19, primals_137, primals_138, buf1266, buf1271, buf1259, buf1264, buf1265, buf1269, buf1274, buf1275, 128, 128, grid=grid(128), stream=stream0)
        del addmm_130
        del buf1256
        del primals_138
        buf1262 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1260, (512, 128), (1, 512), 0), view_352, out=buf1262)
        del view_352
        buf1263 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1260, buf1263, 512, 128, grid=grid(512), stream=stream0)
        buf1268 = reinterpret_tensor(buf1260, (128, 512), (512, 1), 0); del buf1260  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1266, (128, 128), (1, 128), 0), view_350, out=buf1268)
        del view_350
        buf1272 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1270, (512, 128), (1, 512), 0), view_348, out=buf1272)
        del view_348
        buf1273 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1270, buf1273, 512, 128, grid=grid(512), stream=stream0)
        buf1276 = buf1266; del buf1266  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1276, buf1271, primals_137, 16384, grid=grid(16384), stream=stream0)
        del primals_137
        buf1277 = reinterpret_tensor(buf1270, (128, 512), (512, 1), 0); del buf1270  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1276, (128, 128), (128, 1), 0), permute_1553, out=buf1277)
        del permute_1553
        buf1278 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1276, (128, 128), (1, 128), 0), view_346, out=buf1278)
        del view_346
        buf1280 = reinterpret_tensor(buf1277, (1, 128, 512), (65536, 512, 1), 0); del buf1277  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1280, le_64, 65536, grid=grid(65536), stream=stream0)
        del le_64
        buf1281 = buf1271; del buf1271  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1280, (128, 512), (512, 1), 0), permute_1557, out=buf1281)
        del permute_1557
        buf1286 = buf19; del buf19  # reuse
        buf1311 = reinterpret_tensor(buf1261, (1, 128, 128), (16384, 128, 1), 0); del buf1261  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf1276, buf1281, primals_135, primals_131, buf1286, buf1311, 16384, grid=grid(16384), stream=stream0)
        del primals_135
        buf1279 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1284 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1285 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1289 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1309 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1310 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_126, layer_input_44, mul_66], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf1276, buf1281, addmm_126, addmm_121, primals_131, primals_132, buf1286, buf1279, buf1284, buf1285, buf1289, buf1309, buf1310, 128, 128, grid=grid(128), stream=stream0)
        del addmm_121
        del addmm_126
        del primals_131
        del primals_132
        buf1282 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1280, (512, 128), (1, 512), 0), view_344, out=buf1282)
        del view_344
        buf1283 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1280, buf1283, 512, 128, grid=grid(512), stream=stream0)
        buf1287 = buf1281; del buf1281  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1286, (128, 128), (128, 1), 0), permute_1561, out=buf1287)
        del permute_1561
        buf1288 = reinterpret_tensor(buf1276, (128, 128), (128, 1), 0); del buf1276  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1286, (128, 128), (1, 128), 0), view_342, out=buf1288)
        del view_342
        # Source Nodes: [], Original ATen: []
        buf1290 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1287, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_45, clone_default_46, clone_default_47, None, alias_default_31, getitem_156, getitem_157, getitem_158, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_31
        del clone_default_45
        del clone_default_46
        del clone_default_47
        del getitem_156
        del getitem_157
        del getitem_158
        buf1291 = buf1290[0]
        buf1292 = buf1290[1]
        buf1293 = buf1290[2]
        del buf1290
        buf1294 = reinterpret_tensor(buf1280, (128, 512), (512, 1), 0); del buf1280  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1293, (128, 128), (128, 1), 0), permute_1574, out=buf1294)
        del permute_1574
        buf1295 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1293, (128, 128), (1, 128), 0), view_322, out=buf1295)
        buf1296 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1293, buf1296, 128, 128, grid=grid(128), stream=stream0)
        buf1297 = reinterpret_tensor(buf1293, (128, 128), (128, 1), 0); del buf1293  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1292, (128, 128), (128, 1), 0), permute_1578, out=buf1297)
        del permute_1578
        buf1298 = buf1287; del buf1287  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1292, (128, 128), (1, 128), 0), view_326, out=buf1298)
        buf1299 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1292, buf1299, 128, 128, grid=grid(128), stream=stream0)
        buf1300 = reinterpret_tensor(buf1292, (128, 128), (128, 1), 0); del buf1292  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1291, (128, 128), (128, 1), 0), permute_1582, out=buf1300)
        del permute_1582
        buf1301 = reinterpret_tensor(buf1286, (128, 128), (128, 1), 0); del buf1286  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1291, (128, 128), (1, 128), 0), view_326, out=buf1301)
        del view_326
        buf1302 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1291, buf1302, 128, 128, grid=grid(128), stream=stream0)
        del buf1291
        buf1303 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1304 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf1297, buf1300, addmm_122, buf1303, buf1304, 128, 128, grid=grid(128), stream=stream0)
        del addmm_122
        buf1305 = reinterpret_tensor(buf1297, (1, 128, 128), (16384, 128, 1), 0); del buf1297  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1305, buf1300, primals_133, 16384, grid=grid(16384), stream=stream0)
        del primals_133
        buf1306 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1305, (128, 128), (128, 1), 0), permute_1586, out=buf1306)
        del permute_1586
        buf1307 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1305, (128, 128), (1, 128), 0), view_322, out=buf1307)
        buf1308 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1305, buf1308, 128, 128, grid=grid(128), stream=stream0)
        buf1312 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1311, (128, 128), (128, 1), 0), permute_1590, out=buf1312)
        del permute_1590
        buf1313 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1311, (128, 128), (1, 128), 0), view_322, out=buf1313)
        del view_322
        buf1314 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1311, buf1314, 128, 128, grid=grid(128), stream=stream0)
        buf1315 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1317 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_121, mul_57, value_tensor_7], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_28.run(buf1239, buf1294, buf1306, buf1312, addmm_120, buf16, primals_113, primals_114, buf1315, buf1317, 512, 128, grid=grid(512), stream=stream0)
        del addmm_120
        del primals_114
        buf1316 = buf1239; del buf1239  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf1316, buf1294, buf1306, buf1312, primals_129, 65536, grid=grid(65536), stream=stream0)
        del primals_129
        buf1318 = reinterpret_tensor(buf1311, (128, 128), (128, 1), 0); del buf1311  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1316, (128, 512), (512, 1), 0), permute_1594, out=buf1318)
        del permute_1594
        buf1319 = reinterpret_tensor(buf1312, (512, 128), (128, 1), 0); del buf1312  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1316, (512, 128), (1, 512), 0), view_320, out=buf1319)
        del view_320
        buf1320 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf1316, buf1320, 512, 128, grid=grid(512), stream=stream0)
        buf1323 = buf1305; del buf1305  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf1318, primals_127, buf1323, 16384, grid=grid(16384), stream=stream0)
        del primals_127
        buf1324 = buf1306; del buf1306  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1323, (128, 128), (128, 1), 0), permute_1598, out=buf1324)
        del permute_1598
        buf1327 = reinterpret_tensor(buf1324, (1, 128, 512), (65536, 512, 1), 0); del buf1324  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1327, le_65, 65536, grid=grid(65536), stream=stream0)
        del le_65
        buf1328 = buf1300; del buf1300  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1327, (128, 512), (512, 1), 0), permute_1602, out=buf1328)
        del permute_1602
        buf1321 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1322 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1326 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1331 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1332 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_119, attention_output_38, mul_63], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf1318, addmm_119, buf18, primals_125, primals_126, buf1323, buf1328, buf1321, buf1322, buf1326, buf1331, buf1332, 128, 128, grid=grid(128), stream=stream0)
        del addmm_119
        del primals_126
        buf1325 = buf1294; del buf1294  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1323, (128, 128), (1, 128), 0), view_318, out=buf1325)
        del view_318
        buf1329 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1327, (512, 128), (1, 512), 0), view_316, out=buf1329)
        del view_316
        buf1330 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1327, buf1330, 512, 128, grid=grid(512), stream=stream0)
        buf1333 = buf1323; del buf1323  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1333, buf1328, primals_125, 16384, grid=grid(16384), stream=stream0)
        del primals_125
        buf1334 = reinterpret_tensor(buf1327, (128, 512), (512, 1), 0); del buf1327  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1333, (128, 128), (128, 1), 0), permute_1606, out=buf1334)
        del permute_1606
        buf1335 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1333, (128, 128), (1, 128), 0), view_314, out=buf1335)
        del view_314
        buf1337 = reinterpret_tensor(buf1334, (1, 128, 512), (65536, 512, 1), 0); del buf1334  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1337, le_66, 65536, grid=grid(65536), stream=stream0)
        del le_66
        buf1338 = buf1328; del buf1328  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1337, (128, 512), (512, 1), 0), permute_1610, out=buf1338)
        del permute_1610
        buf1343 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf1333, buf1338, primals_123, buf1343, 16384, grid=grid(16384), stream=stream0)
        del primals_123
        buf1344 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1343, (128, 128), (128, 1), 0), permute_1614, out=buf1344)
        del permute_1614
        buf1347 = reinterpret_tensor(buf1344, (1, 128, 512), (65536, 512, 1), 0); del buf1344  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1347, le_67, 65536, grid=grid(65536), stream=stream0)
        del le_67
        buf1348 = buf1318; del buf1318  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1347, (128, 512), (512, 1), 0), permute_1618, out=buf1348)
        del permute_1618
        buf1336 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1341 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1342 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1346 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1351 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1352 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_115, attention_output_36, mul_61], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf1333, buf1338, addmm_115, buf17, primals_121, primals_122, buf1343, buf1348, buf1336, buf1341, buf1342, buf1346, buf1351, buf1352, 128, 128, grid=grid(128), stream=stream0)
        del addmm_115
        del buf1333
        del primals_122
        buf1339 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1337, (512, 128), (1, 512), 0), view_312, out=buf1339)
        del view_312
        buf1340 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1337, buf1340, 512, 128, grid=grid(512), stream=stream0)
        buf1345 = reinterpret_tensor(buf1337, (128, 512), (512, 1), 0); del buf1337  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1343, (128, 128), (1, 128), 0), view_310, out=buf1345)
        del view_310
        buf1349 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1347, (512, 128), (1, 512), 0), view_308, out=buf1349)
        del view_308
        buf1350 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1347, buf1350, 512, 128, grid=grid(512), stream=stream0)
        buf1353 = buf1343; del buf1343  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1353, buf1348, primals_121, 16384, grid=grid(16384), stream=stream0)
        del primals_121
        buf1354 = reinterpret_tensor(buf1347, (128, 512), (512, 1), 0); del buf1347  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1353, (128, 128), (128, 1), 0), permute_1622, out=buf1354)
        del permute_1622
        buf1355 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1353, (128, 128), (1, 128), 0), view_306, out=buf1355)
        del view_306
        buf1357 = reinterpret_tensor(buf1354, (1, 128, 512), (65536, 512, 1), 0); del buf1354  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1357, le_68, 65536, grid=grid(65536), stream=stream0)
        del le_68
        buf1358 = buf1348; del buf1348  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1357, (128, 512), (512, 1), 0), permute_1626, out=buf1358)
        del permute_1626
        buf1363 = buf17; del buf17  # reuse
        buf1388 = reinterpret_tensor(buf1338, (1, 128, 128), (16384, 128, 1), 0); del buf1338  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf1353, buf1358, primals_119, primals_115, buf1363, buf1388, 16384, grid=grid(16384), stream=stream0)
        del primals_119
        buf1356 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1361 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1362 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1366 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1386 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1387 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_111, layer_input_39, mul_58], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf1353, buf1358, addmm_111, addmm_106, primals_115, primals_116, buf1363, buf1356, buf1361, buf1362, buf1366, buf1386, buf1387, 128, 128, grid=grid(128), stream=stream0)
        del addmm_106
        del addmm_111
        del primals_115
        del primals_116
        buf1359 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1357, (512, 128), (1, 512), 0), view_304, out=buf1359)
        del view_304
        buf1360 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1357, buf1360, 512, 128, grid=grid(512), stream=stream0)
        buf1364 = buf1358; del buf1358  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1363, (128, 128), (128, 1), 0), permute_1630, out=buf1364)
        del permute_1630
        buf1365 = reinterpret_tensor(buf1353, (128, 128), (128, 1), 0); del buf1353  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1363, (128, 128), (1, 128), 0), view_302, out=buf1365)
        del view_302
        # Source Nodes: [], Original ATen: []
        buf1367 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1364, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_48, clone_default_49, clone_default_50, None, alias_default_33, getitem_163, getitem_164, getitem_165, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_33
        del clone_default_48
        del clone_default_49
        del clone_default_50
        del getitem_163
        del getitem_164
        del getitem_165
        buf1368 = buf1367[0]
        buf1369 = buf1367[1]
        buf1370 = buf1367[2]
        del buf1367
        buf1371 = reinterpret_tensor(buf1357, (128, 512), (512, 1), 0); del buf1357  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1370, (128, 128), (128, 1), 0), permute_1643, out=buf1371)
        del permute_1643
        buf1372 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1370, (128, 128), (1, 128), 0), view_282, out=buf1372)
        buf1373 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1370, buf1373, 128, 128, grid=grid(128), stream=stream0)
        buf1374 = reinterpret_tensor(buf1370, (128, 128), (128, 1), 0); del buf1370  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1369, (128, 128), (128, 1), 0), permute_1647, out=buf1374)
        del permute_1647
        buf1375 = buf1364; del buf1364  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1369, (128, 128), (1, 128), 0), view_286, out=buf1375)
        buf1376 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1369, buf1376, 128, 128, grid=grid(128), stream=stream0)
        buf1377 = reinterpret_tensor(buf1369, (128, 128), (128, 1), 0); del buf1369  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1368, (128, 128), (128, 1), 0), permute_1651, out=buf1377)
        del permute_1651
        buf1378 = reinterpret_tensor(buf1363, (128, 128), (128, 1), 0); del buf1363  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1368, (128, 128), (1, 128), 0), view_286, out=buf1378)
        del view_286
        buf1379 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1368, buf1379, 128, 128, grid=grid(128), stream=stream0)
        del buf1368
        buf1380 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1381 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf1374, buf1377, addmm_107, buf1380, buf1381, 128, 128, grid=grid(128), stream=stream0)
        del addmm_107
        buf1382 = reinterpret_tensor(buf1374, (1, 128, 128), (16384, 128, 1), 0); del buf1374  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1382, buf1377, primals_117, 16384, grid=grid(16384), stream=stream0)
        del primals_117
        buf1383 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1382, (128, 128), (128, 1), 0), permute_1655, out=buf1383)
        del permute_1655
        buf1384 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1382, (128, 128), (1, 128), 0), view_282, out=buf1384)
        buf1385 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1382, buf1385, 128, 128, grid=grid(128), stream=stream0)
        buf1389 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1388, (128, 128), (128, 1), 0), permute_1659, out=buf1389)
        del permute_1659
        buf1390 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1388, (128, 128), (1, 128), 0), view_282, out=buf1390)
        del view_282
        buf1391 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1388, buf1391, 128, 128, grid=grid(128), stream=stream0)
        buf1392 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1394 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_26.run(buf1316, buf1371, buf1383, buf1389, buf16, buf1392, buf1394, 512, 128, grid=grid(512), stream=stream0)
        buf1393 = buf1316; del buf1316  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf1393, buf1371, buf1383, buf1389, primals_113, 65536, grid=grid(65536), stream=stream0)
        del primals_113
        buf1395 = reinterpret_tensor(buf1388, (128, 128), (128, 1), 0); del buf1388  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1393, (128, 512), (512, 1), 0), permute_1663, out=buf1395)
        del permute_1663
        buf1396 = reinterpret_tensor(buf1389, (512, 128), (128, 1), 0); del buf1389  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1393, (512, 128), (1, 512), 0), view_280, out=buf1396)
        del view_280
        buf1397 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf1393, buf1397, 512, 128, grid=grid(512), stream=stream0)
        buf1400 = buf1382; del buf1382  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf1395, primals_111, buf1400, 16384, grid=grid(16384), stream=stream0)
        del primals_111
        buf1401 = buf1383; del buf1383  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1400, (128, 128), (128, 1), 0), permute_1667, out=buf1401)
        del permute_1667
        buf1404 = reinterpret_tensor(buf1401, (1, 128, 512), (65536, 512, 1), 0); del buf1401  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1404, le_69, 65536, grid=grid(65536), stream=stream0)
        del le_69
        buf1405 = buf1377; del buf1377  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1404, (128, 512), (512, 1), 0), permute_1671, out=buf1405)
        del permute_1671
        buf1398 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1399 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1403 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1408 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1409 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_104, attention_output_33, mul_55], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf1395, addmm_104, buf15, primals_109, primals_110, buf1400, buf1405, buf1398, buf1399, buf1403, buf1408, buf1409, 128, 128, grid=grid(128), stream=stream0)
        del addmm_104
        del primals_110
        buf1402 = buf1371; del buf1371  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1400, (128, 128), (1, 128), 0), view_278, out=buf1402)
        del view_278
        buf1406 = reinterpret_tensor(buf16, (512, 128), (128, 1), 0); del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1404, (512, 128), (1, 512), 0), view_276, out=buf1406)
        del view_276
        buf1407 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1404, buf1407, 512, 128, grid=grid(512), stream=stream0)
        buf1410 = buf1400; del buf1400  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1410, buf1405, primals_109, 16384, grid=grid(16384), stream=stream0)
        del primals_109
        buf1411 = reinterpret_tensor(buf1404, (128, 512), (512, 1), 0); del buf1404  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1410, (128, 128), (128, 1), 0), permute_1675, out=buf1411)
        del permute_1675
        buf1412 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1410, (128, 128), (1, 128), 0), view_274, out=buf1412)
        del view_274
        buf1414 = reinterpret_tensor(buf1411, (1, 128, 512), (65536, 512, 1), 0); del buf1411  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1414, le_70, 65536, grid=grid(65536), stream=stream0)
        del le_70
        buf1415 = buf1405; del buf1405  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1414, (128, 512), (512, 1), 0), permute_1679, out=buf1415)
        del permute_1679
        buf1420 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf1410, buf1415, primals_107, buf1420, 16384, grid=grid(16384), stream=stream0)
        del primals_107
        buf1421 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1420, (128, 128), (128, 1), 0), permute_1683, out=buf1421)
        del permute_1683
        buf1424 = reinterpret_tensor(buf1421, (1, 128, 512), (65536, 512, 1), 0); del buf1421  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1424, le_71, 65536, grid=grid(65536), stream=stream0)
        del le_71
        buf1425 = buf1395; del buf1395  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1424, (128, 512), (512, 1), 0), permute_1687, out=buf1425)
        del permute_1687
        buf1413 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1418 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1419 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1423 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1428 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1429 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_100, attention_output_31, mul_53], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf1410, buf1415, addmm_100, buf14, primals_105, primals_106, buf1420, buf1425, buf1413, buf1418, buf1419, buf1423, buf1428, buf1429, 128, 128, grid=grid(128), stream=stream0)
        del addmm_100
        del buf14
        del primals_106
        buf1416 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1414, (512, 128), (1, 512), 0), view_272, out=buf1416)
        del view_272
        buf1417 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1414, buf1417, 512, 128, grid=grid(512), stream=stream0)
        buf1422 = reinterpret_tensor(buf1414, (128, 512), (512, 1), 0); del buf1414  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1420, (128, 128), (1, 128), 0), view_270, out=buf1422)
        del view_270
        buf1426 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1424, (512, 128), (1, 512), 0), view_268, out=buf1426)
        del view_268
        buf1427 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1424, buf1427, 512, 128, grid=grid(512), stream=stream0)
        buf1430 = buf1420; del buf1420  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1430, buf1425, primals_105, 16384, grid=grid(16384), stream=stream0)
        del primals_105
        buf1431 = reinterpret_tensor(buf1424, (128, 512), (512, 1), 0); del buf1424  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1430, (128, 128), (128, 1), 0), permute_1691, out=buf1431)
        del permute_1691
        buf1432 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1430, (128, 128), (1, 128), 0), view_266, out=buf1432)
        del view_266
        buf1434 = reinterpret_tensor(buf1431, (1, 128, 512), (65536, 512, 1), 0); del buf1431  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1434, le_72, 65536, grid=grid(65536), stream=stream0)
        del le_72
        buf1435 = buf1425; del buf1425  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1434, (128, 512), (512, 1), 0), permute_1695, out=buf1435)
        del permute_1695
        buf1440 = reinterpret_tensor(buf1415, (1, 128, 128), (16384, 128, 1), 0); del buf1415  # reuse
        buf1465 = buf1410; del buf1410  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf1430, buf1435, primals_103, primals_99, buf1440, buf1465, 16384, grid=grid(16384), stream=stream0)
        del primals_103
        buf1433 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1438 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1439 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1443 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1463 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1464 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_96, layer_input_34, mul_50], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf1430, buf1435, addmm_96, addmm_91, primals_99, primals_100, buf1440, buf1433, buf1438, buf1439, buf1443, buf1463, buf1464, 128, 128, grid=grid(128), stream=stream0)
        del addmm_91
        del addmm_96
        del primals_100
        del primals_99
        buf1436 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1434, (512, 128), (1, 512), 0), view_264, out=buf1436)
        del view_264
        buf1437 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1434, buf1437, 512, 128, grid=grid(512), stream=stream0)
        buf1441 = buf1435; del buf1435  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1440, (128, 128), (128, 1), 0), permute_1699, out=buf1441)
        del permute_1699
        buf1442 = reinterpret_tensor(buf1430, (128, 128), (128, 1), 0); del buf1430  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1440, (128, 128), (1, 128), 0), view_262, out=buf1442)
        del view_262
        # Source Nodes: [], Original ATen: []
        buf1444 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1441, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_51, clone_default_52, clone_default_53, None, alias_default_35, getitem_170, getitem_171, getitem_172, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_35
        del clone_default_51
        del clone_default_52
        del clone_default_53
        del getitem_170
        del getitem_171
        del getitem_172
        buf1445 = buf1444[0]
        buf1446 = buf1444[1]
        buf1447 = buf1444[2]
        del buf1444
        buf1448 = reinterpret_tensor(buf1434, (128, 512), (512, 1), 0); del buf1434  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1447, (128, 128), (128, 1), 0), permute_1712, out=buf1448)
        del permute_1712
        buf1449 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1447, (128, 128), (1, 128), 0), view_242, out=buf1449)
        buf1450 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1447, buf1450, 128, 128, grid=grid(128), stream=stream0)
        buf1451 = reinterpret_tensor(buf1447, (128, 128), (128, 1), 0); del buf1447  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1446, (128, 128), (128, 1), 0), permute_1716, out=buf1451)
        del permute_1716
        buf1452 = buf1441; del buf1441  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1446, (128, 128), (1, 128), 0), view_246, out=buf1452)
        buf1453 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1446, buf1453, 128, 128, grid=grid(128), stream=stream0)
        buf1454 = reinterpret_tensor(buf1446, (128, 128), (128, 1), 0); del buf1446  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1445, (128, 128), (128, 1), 0), permute_1720, out=buf1454)
        del permute_1720
        buf1455 = reinterpret_tensor(buf1440, (128, 128), (128, 1), 0); del buf1440  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1445, (128, 128), (1, 128), 0), view_246, out=buf1455)
        del view_246
        buf1456 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1445, buf1456, 128, 128, grid=grid(128), stream=stream0)
        del buf1445
        buf1457 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1458 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf1451, buf1454, addmm_92, buf1457, buf1458, 128, 128, grid=grid(128), stream=stream0)
        del addmm_92
        buf1459 = reinterpret_tensor(buf1451, (1, 128, 128), (16384, 128, 1), 0); del buf1451  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1459, buf1454, primals_101, 16384, grid=grid(16384), stream=stream0)
        del primals_101
        buf1460 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1459, (128, 128), (128, 1), 0), permute_1724, out=buf1460)
        del permute_1724
        buf1461 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1459, (128, 128), (1, 128), 0), view_242, out=buf1461)
        buf1462 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1459, buf1462, 128, 128, grid=grid(128), stream=stream0)
        buf1466 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1465, (128, 128), (128, 1), 0), permute_1728, out=buf1466)
        del permute_1728
        buf1467 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1465, (128, 128), (1, 128), 0), view_242, out=buf1467)
        del view_242
        buf1468 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1465, buf1468, 128, 128, grid=grid(128), stream=stream0)
        buf1469 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1471 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_91, mul_41, value_tensor_5], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_28.run(buf1393, buf1448, buf1460, buf1466, addmm_90, buf11, primals_81, primals_82, buf1469, buf1471, 512, 128, grid=grid(512), stream=stream0)
        del addmm_90
        del primals_82
        buf1470 = buf1393; del buf1393  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf1470, buf1448, buf1460, buf1466, primals_97, 65536, grid=grid(65536), stream=stream0)
        del primals_97
        buf1472 = reinterpret_tensor(buf1465, (128, 128), (128, 1), 0); del buf1465  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1470, (128, 512), (512, 1), 0), permute_1732, out=buf1472)
        del permute_1732
        buf1473 = reinterpret_tensor(buf1466, (512, 128), (128, 1), 0); del buf1466  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1470, (512, 128), (1, 512), 0), view_240, out=buf1473)
        del view_240
        buf1474 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf1470, buf1474, 512, 128, grid=grid(512), stream=stream0)
        buf1477 = buf1459; del buf1459  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf1472, primals_95, buf1477, 16384, grid=grid(16384), stream=stream0)
        del primals_95
        buf1478 = buf1460; del buf1460  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1477, (128, 128), (128, 1), 0), permute_1736, out=buf1478)
        del permute_1736
        buf1481 = reinterpret_tensor(buf1478, (1, 128, 512), (65536, 512, 1), 0); del buf1478  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1481, le_73, 65536, grid=grid(65536), stream=stream0)
        del le_73
        buf1482 = buf1454; del buf1454  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1481, (128, 512), (512, 1), 0), permute_1740, out=buf1482)
        del permute_1740
        buf1475 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1476 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1480 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1485 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1486 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_89, attention_output_28, mul_47], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf1472, addmm_89, buf13, primals_93, primals_94, buf1477, buf1482, buf1475, buf1476, buf1480, buf1485, buf1486, 128, 128, grid=grid(128), stream=stream0)
        del addmm_89
        del primals_94
        buf1479 = buf1448; del buf1448  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1477, (128, 128), (1, 128), 0), view_238, out=buf1479)
        del view_238
        buf1483 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1481, (512, 128), (1, 512), 0), view_236, out=buf1483)
        del view_236
        buf1484 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1481, buf1484, 512, 128, grid=grid(512), stream=stream0)
        buf1487 = buf1477; del buf1477  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1487, buf1482, primals_93, 16384, grid=grid(16384), stream=stream0)
        del primals_93
        buf1488 = reinterpret_tensor(buf1481, (128, 512), (512, 1), 0); del buf1481  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1487, (128, 128), (128, 1), 0), permute_1744, out=buf1488)
        del permute_1744
        buf1489 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1487, (128, 128), (1, 128), 0), view_234, out=buf1489)
        del view_234
        buf1491 = reinterpret_tensor(buf1488, (1, 128, 512), (65536, 512, 1), 0); del buf1488  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1491, le_74, 65536, grid=grid(65536), stream=stream0)
        del le_74
        buf1492 = buf1482; del buf1482  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1491, (128, 512), (512, 1), 0), permute_1748, out=buf1492)
        del permute_1748
        buf1497 = reinterpret_tensor(buf1472, (1, 128, 128), (16384, 128, 1), 0); del buf1472  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf1487, buf1492, primals_91, buf1497, 16384, grid=grid(16384), stream=stream0)
        del primals_91
        buf1498 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1497, (128, 128), (128, 1), 0), permute_1752, out=buf1498)
        del permute_1752
        buf1501 = reinterpret_tensor(buf1498, (1, 128, 512), (65536, 512, 1), 0); del buf1498  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1501, le_75, 65536, grid=grid(65536), stream=stream0)
        del le_75
        buf1502 = reinterpret_tensor(buf13, (128, 128), (128, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1501, (128, 512), (512, 1), 0), permute_1756, out=buf1502)
        del permute_1756
        buf1490 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1495 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1496 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1500 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1505 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1506 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_85, attention_output_26, mul_45], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf1487, buf1492, addmm_85, buf12, primals_89, primals_90, buf1497, buf1502, buf1490, buf1495, buf1496, buf1500, buf1505, buf1506, 128, 128, grid=grid(128), stream=stream0)
        del addmm_85
        del buf12
        del primals_90
        buf1493 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1491, (512, 128), (1, 512), 0), view_232, out=buf1493)
        del view_232
        buf1494 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1491, buf1494, 512, 128, grid=grid(512), stream=stream0)
        buf1499 = reinterpret_tensor(buf1491, (128, 512), (512, 1), 0); del buf1491  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1497, (128, 128), (1, 128), 0), view_230, out=buf1499)
        del view_230
        buf1503 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1501, (512, 128), (1, 512), 0), view_228, out=buf1503)
        del view_228
        buf1504 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1501, buf1504, 512, 128, grid=grid(512), stream=stream0)
        buf1507 = buf1497; del buf1497  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1507, buf1502, primals_89, 16384, grid=grid(16384), stream=stream0)
        del primals_89
        buf1508 = reinterpret_tensor(buf1501, (128, 512), (512, 1), 0); del buf1501  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1507, (128, 128), (128, 1), 0), permute_1760, out=buf1508)
        del permute_1760
        buf1509 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1507, (128, 128), (1, 128), 0), view_226, out=buf1509)
        del view_226
        buf1511 = reinterpret_tensor(buf1508, (1, 128, 512), (65536, 512, 1), 0); del buf1508  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1511, le_76, 65536, grid=grid(65536), stream=stream0)
        del le_76
        buf1512 = buf1502; del buf1502  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1511, (128, 512), (512, 1), 0), permute_1764, out=buf1512)
        del permute_1764
        buf1517 = reinterpret_tensor(buf1492, (1, 128, 128), (16384, 128, 1), 0); del buf1492  # reuse
        buf1542 = buf1487; del buf1487  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf1507, buf1512, primals_87, primals_83, buf1517, buf1542, 16384, grid=grid(16384), stream=stream0)
        del primals_87
        buf1510 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1515 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1516 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1520 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1540 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1541 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_81, layer_input_29, mul_42], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf1507, buf1512, addmm_81, addmm_76, primals_83, primals_84, buf1517, buf1510, buf1515, buf1516, buf1520, buf1540, buf1541, 128, 128, grid=grid(128), stream=stream0)
        del addmm_76
        del addmm_81
        del primals_83
        del primals_84
        buf1513 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1511, (512, 128), (1, 512), 0), view_224, out=buf1513)
        del view_224
        buf1514 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1511, buf1514, 512, 128, grid=grid(512), stream=stream0)
        buf1518 = buf1512; del buf1512  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1517, (128, 128), (128, 1), 0), permute_1768, out=buf1518)
        del permute_1768
        buf1519 = reinterpret_tensor(buf1507, (128, 128), (128, 1), 0); del buf1507  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1517, (128, 128), (1, 128), 0), view_222, out=buf1519)
        del view_222
        # Source Nodes: [], Original ATen: []
        buf1521 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1518, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_54, clone_default_55, clone_default_56, None, alias_default_37, getitem_177, getitem_178, getitem_179, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_37
        del clone_default_54
        del clone_default_55
        del clone_default_56
        del getitem_177
        del getitem_178
        del getitem_179
        buf1522 = buf1521[0]
        buf1523 = buf1521[1]
        buf1524 = buf1521[2]
        del buf1521
        buf1525 = reinterpret_tensor(buf1511, (128, 512), (512, 1), 0); del buf1511  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1524, (128, 128), (128, 1), 0), permute_1781, out=buf1525)
        del permute_1781
        buf1526 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1524, (128, 128), (1, 128), 0), view_202, out=buf1526)
        buf1527 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1524, buf1527, 128, 128, grid=grid(128), stream=stream0)
        buf1528 = reinterpret_tensor(buf1524, (128, 128), (128, 1), 0); del buf1524  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1523, (128, 128), (128, 1), 0), permute_1785, out=buf1528)
        del permute_1785
        buf1529 = buf1518; del buf1518  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1523, (128, 128), (1, 128), 0), view_206, out=buf1529)
        buf1530 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1523, buf1530, 128, 128, grid=grid(128), stream=stream0)
        buf1531 = reinterpret_tensor(buf1523, (128, 128), (128, 1), 0); del buf1523  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1522, (128, 128), (128, 1), 0), permute_1789, out=buf1531)
        del permute_1789
        buf1532 = reinterpret_tensor(buf1517, (128, 128), (128, 1), 0); del buf1517  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1522, (128, 128), (1, 128), 0), view_206, out=buf1532)
        del view_206
        buf1533 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1522, buf1533, 128, 128, grid=grid(128), stream=stream0)
        del buf1522
        buf1534 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1535 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf1528, buf1531, addmm_77, buf1534, buf1535, 128, 128, grid=grid(128), stream=stream0)
        del addmm_77
        buf1536 = reinterpret_tensor(buf1528, (1, 128, 128), (16384, 128, 1), 0); del buf1528  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1536, buf1531, primals_85, 16384, grid=grid(16384), stream=stream0)
        del primals_85
        buf1537 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1536, (128, 128), (128, 1), 0), permute_1793, out=buf1537)
        del permute_1793
        buf1538 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1536, (128, 128), (1, 128), 0), view_202, out=buf1538)
        buf1539 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1536, buf1539, 128, 128, grid=grid(128), stream=stream0)
        buf1543 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1542, (128, 128), (128, 1), 0), permute_1797, out=buf1543)
        del permute_1797
        buf1544 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1542, (128, 128), (1, 128), 0), view_202, out=buf1544)
        del view_202
        buf1545 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1542, buf1545, 128, 128, grid=grid(128), stream=stream0)
        buf1546 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1548 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_26.run(buf1470, buf1525, buf1537, buf1543, buf11, buf1546, buf1548, 512, 128, grid=grid(512), stream=stream0)
        buf1547 = buf1470; del buf1470  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf1547, buf1525, buf1537, buf1543, primals_81, 65536, grid=grid(65536), stream=stream0)
        del primals_81
        buf1549 = reinterpret_tensor(buf1542, (128, 128), (128, 1), 0); del buf1542  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1547, (128, 512), (512, 1), 0), permute_1801, out=buf1549)
        del permute_1801
        buf1550 = reinterpret_tensor(buf1543, (512, 128), (128, 1), 0); del buf1543  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1547, (512, 128), (1, 512), 0), view_200, out=buf1550)
        del view_200
        buf1551 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf1547, buf1551, 512, 128, grid=grid(512), stream=stream0)
        buf1554 = buf1536; del buf1536  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf1549, primals_79, buf1554, 16384, grid=grid(16384), stream=stream0)
        del primals_79
        buf1555 = buf1537; del buf1537  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1554, (128, 128), (128, 1), 0), permute_1805, out=buf1555)
        del permute_1805
        buf1558 = reinterpret_tensor(buf1555, (1, 128, 512), (65536, 512, 1), 0); del buf1555  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1558, le_77, 65536, grid=grid(65536), stream=stream0)
        del le_77
        buf1559 = buf1531; del buf1531  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1558, (128, 512), (512, 1), 0), permute_1809, out=buf1559)
        del permute_1809
        buf1552 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1553 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1557 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1562 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1563 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_74, attention_output_23, mul_39], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf1549, addmm_74, buf10, primals_77, primals_78, buf1554, buf1559, buf1552, buf1553, buf1557, buf1562, buf1563, 128, 128, grid=grid(128), stream=stream0)
        del addmm_74
        del primals_78
        buf1556 = buf1525; del buf1525  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1554, (128, 128), (1, 128), 0), view_198, out=buf1556)
        del view_198
        buf1560 = reinterpret_tensor(buf11, (512, 128), (128, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1558, (512, 128), (1, 512), 0), view_196, out=buf1560)
        del view_196
        buf1561 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1558, buf1561, 512, 128, grid=grid(512), stream=stream0)
        buf1564 = buf1554; del buf1554  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1564, buf1559, primals_77, 16384, grid=grid(16384), stream=stream0)
        del primals_77
        buf1565 = reinterpret_tensor(buf1558, (128, 512), (512, 1), 0); del buf1558  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1564, (128, 128), (128, 1), 0), permute_1813, out=buf1565)
        del permute_1813
        buf1566 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1564, (128, 128), (1, 128), 0), view_194, out=buf1566)
        del view_194
        buf1568 = reinterpret_tensor(buf1565, (1, 128, 512), (65536, 512, 1), 0); del buf1565  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1568, le_78, 65536, grid=grid(65536), stream=stream0)
        del le_78
        buf1569 = buf1559; del buf1559  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1568, (128, 512), (512, 1), 0), permute_1817, out=buf1569)
        del permute_1817
        buf1574 = reinterpret_tensor(buf1549, (1, 128, 128), (16384, 128, 1), 0); del buf1549  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf1564, buf1569, primals_75, buf1574, 16384, grid=grid(16384), stream=stream0)
        del primals_75
        buf1575 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1574, (128, 128), (128, 1), 0), permute_1821, out=buf1575)
        del permute_1821
        buf1578 = reinterpret_tensor(buf1575, (1, 128, 512), (65536, 512, 1), 0); del buf1575  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1578, le_79, 65536, grid=grid(65536), stream=stream0)
        del le_79
        buf1579 = reinterpret_tensor(buf10, (128, 128), (128, 1), 0); del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1578, (128, 512), (512, 1), 0), permute_1825, out=buf1579)
        del permute_1825
        buf1567 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1572 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1573 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1577 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1582 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1583 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_70, attention_output_21, mul_37], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf1564, buf1569, addmm_70, buf9, primals_73, primals_74, buf1574, buf1579, buf1567, buf1572, buf1573, buf1577, buf1582, buf1583, 128, 128, grid=grid(128), stream=stream0)
        del addmm_70
        del buf1564
        del primals_74
        buf1570 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1568, (512, 128), (1, 512), 0), view_192, out=buf1570)
        del view_192
        buf1571 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1568, buf1571, 512, 128, grid=grid(512), stream=stream0)
        buf1576 = reinterpret_tensor(buf1568, (128, 512), (512, 1), 0); del buf1568  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1574, (128, 128), (1, 128), 0), view_190, out=buf1576)
        del view_190
        buf1580 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1578, (512, 128), (1, 512), 0), view_188, out=buf1580)
        del view_188
        buf1581 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1578, buf1581, 512, 128, grid=grid(512), stream=stream0)
        buf1584 = buf1574; del buf1574  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1584, buf1579, primals_73, 16384, grid=grid(16384), stream=stream0)
        del primals_73
        buf1585 = reinterpret_tensor(buf1578, (128, 512), (512, 1), 0); del buf1578  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1584, (128, 128), (128, 1), 0), permute_1829, out=buf1585)
        del permute_1829
        buf1586 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1584, (128, 128), (1, 128), 0), view_186, out=buf1586)
        del view_186
        buf1588 = reinterpret_tensor(buf1585, (1, 128, 512), (65536, 512, 1), 0); del buf1585  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1588, le_80, 65536, grid=grid(65536), stream=stream0)
        del le_80
        buf1589 = buf1579; del buf1579  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1588, (128, 512), (512, 1), 0), permute_1833, out=buf1589)
        del permute_1833
        buf1594 = buf9; del buf9  # reuse
        buf1619 = reinterpret_tensor(buf1569, (1, 128, 128), (16384, 128, 1), 0); del buf1569  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf1584, buf1589, primals_71, primals_67, buf1594, buf1619, 16384, grid=grid(16384), stream=stream0)
        del primals_71
        buf1587 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1592 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1593 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1597 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1617 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1618 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_66, layer_input_24, mul_34], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf1584, buf1589, addmm_66, addmm_61, primals_67, primals_68, buf1594, buf1587, buf1592, buf1593, buf1597, buf1617, buf1618, 128, 128, grid=grid(128), stream=stream0)
        del addmm_61
        del addmm_66
        del primals_67
        del primals_68
        buf1590 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1588, (512, 128), (1, 512), 0), view_184, out=buf1590)
        del view_184
        buf1591 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1588, buf1591, 512, 128, grid=grid(512), stream=stream0)
        buf1595 = buf1589; del buf1589  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1594, (128, 128), (128, 1), 0), permute_1837, out=buf1595)
        del permute_1837
        buf1596 = reinterpret_tensor(buf1584, (128, 128), (128, 1), 0); del buf1584  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1594, (128, 128), (1, 128), 0), view_182, out=buf1596)
        del view_182
        # Source Nodes: [], Original ATen: []
        buf1598 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1595, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_57, clone_default_58, clone_default_59, None, alias_default_39, getitem_184, getitem_185, getitem_186, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_39
        del clone_default_57
        del clone_default_58
        del clone_default_59
        del getitem_184
        del getitem_185
        del getitem_186
        buf1599 = buf1598[0]
        buf1600 = buf1598[1]
        buf1601 = buf1598[2]
        del buf1598
        buf1602 = reinterpret_tensor(buf1588, (128, 512), (512, 1), 0); del buf1588  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1601, (128, 128), (128, 1), 0), permute_1850, out=buf1602)
        del permute_1850
        buf1603 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1601, (128, 128), (1, 128), 0), view_162, out=buf1603)
        buf1604 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1601, buf1604, 128, 128, grid=grid(128), stream=stream0)
        buf1605 = reinterpret_tensor(buf1601, (128, 128), (128, 1), 0); del buf1601  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1600, (128, 128), (128, 1), 0), permute_1854, out=buf1605)
        del permute_1854
        buf1606 = buf1595; del buf1595  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1600, (128, 128), (1, 128), 0), view_166, out=buf1606)
        buf1607 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1600, buf1607, 128, 128, grid=grid(128), stream=stream0)
        buf1608 = reinterpret_tensor(buf1600, (128, 128), (128, 1), 0); del buf1600  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1599, (128, 128), (128, 1), 0), permute_1858, out=buf1608)
        del permute_1858
        buf1609 = reinterpret_tensor(buf1594, (128, 128), (128, 1), 0); del buf1594  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1599, (128, 128), (1, 128), 0), view_166, out=buf1609)
        del view_166
        buf1610 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1599, buf1610, 128, 128, grid=grid(128), stream=stream0)
        del buf1599
        buf1611 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1612 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf1605, buf1608, addmm_62, buf1611, buf1612, 128, 128, grid=grid(128), stream=stream0)
        del addmm_62
        buf1613 = reinterpret_tensor(buf1605, (1, 128, 128), (16384, 128, 1), 0); del buf1605  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1613, buf1608, primals_69, 16384, grid=grid(16384), stream=stream0)
        del primals_69
        buf1614 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1613, (128, 128), (128, 1), 0), permute_1862, out=buf1614)
        del permute_1862
        buf1615 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1613, (128, 128), (1, 128), 0), view_162, out=buf1615)
        buf1616 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1613, buf1616, 128, 128, grid=grid(128), stream=stream0)
        buf1620 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1619, (128, 128), (128, 1), 0), permute_1866, out=buf1620)
        del permute_1866
        buf1621 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1619, (128, 128), (1, 128), 0), view_162, out=buf1621)
        del view_162
        buf1622 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1619, buf1622, 128, 128, grid=grid(128), stream=stream0)
        buf1623 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1625 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_61, mul_25, value_tensor_3], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_28.run(buf1547, buf1602, buf1614, buf1620, addmm_60, buf6, primals_49, primals_50, buf1623, buf1625, 512, 128, grid=grid(512), stream=stream0)
        del addmm_60
        del primals_50
        buf1624 = buf1547; del buf1547  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf1624, buf1602, buf1614, buf1620, primals_65, 65536, grid=grid(65536), stream=stream0)
        del primals_65
        buf1626 = reinterpret_tensor(buf1619, (128, 128), (128, 1), 0); del buf1619  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1624, (128, 512), (512, 1), 0), permute_1870, out=buf1626)
        del permute_1870
        buf1627 = reinterpret_tensor(buf1620, (512, 128), (128, 1), 0); del buf1620  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1624, (512, 128), (1, 512), 0), view_160, out=buf1627)
        del view_160
        buf1628 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf1624, buf1628, 512, 128, grid=grid(512), stream=stream0)
        buf1631 = buf1613; del buf1613  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf1626, primals_63, buf1631, 16384, grid=grid(16384), stream=stream0)
        del primals_63
        buf1632 = buf1614; del buf1614  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1631, (128, 128), (128, 1), 0), permute_1874, out=buf1632)
        del permute_1874
        buf1635 = reinterpret_tensor(buf1632, (1, 128, 512), (65536, 512, 1), 0); del buf1632  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1635, le_81, 65536, grid=grid(65536), stream=stream0)
        del le_81
        buf1636 = buf1608; del buf1608  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1635, (128, 512), (512, 1), 0), permute_1878, out=buf1636)
        del permute_1878
        buf1629 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1630 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1634 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1639 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1640 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_59, attention_output_18, mul_31], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf1626, addmm_59, buf8, primals_61, primals_62, buf1631, buf1636, buf1629, buf1630, buf1634, buf1639, buf1640, 128, 128, grid=grid(128), stream=stream0)
        del addmm_59
        del primals_62
        buf1633 = buf1602; del buf1602  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1631, (128, 128), (1, 128), 0), view_158, out=buf1633)
        del view_158
        buf1637 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1635, (512, 128), (1, 512), 0), view_156, out=buf1637)
        del view_156
        buf1638 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1635, buf1638, 512, 128, grid=grid(512), stream=stream0)
        buf1641 = buf1631; del buf1631  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1641, buf1636, primals_61, 16384, grid=grid(16384), stream=stream0)
        del primals_61
        buf1642 = reinterpret_tensor(buf1635, (128, 512), (512, 1), 0); del buf1635  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1641, (128, 128), (128, 1), 0), permute_1882, out=buf1642)
        del permute_1882
        buf1643 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1641, (128, 128), (1, 128), 0), view_154, out=buf1643)
        del view_154
        buf1645 = reinterpret_tensor(buf1642, (1, 128, 512), (65536, 512, 1), 0); del buf1642  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1645, le_82, 65536, grid=grid(65536), stream=stream0)
        del le_82
        buf1646 = buf1636; del buf1636  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1645, (128, 512), (512, 1), 0), permute_1886, out=buf1646)
        del permute_1886
        buf1651 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf1641, buf1646, primals_59, buf1651, 16384, grid=grid(16384), stream=stream0)
        del primals_59
        buf1652 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1651, (128, 128), (128, 1), 0), permute_1890, out=buf1652)
        del permute_1890
        buf1655 = reinterpret_tensor(buf1652, (1, 128, 512), (65536, 512, 1), 0); del buf1652  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1655, le_83, 65536, grid=grid(65536), stream=stream0)
        del le_83
        buf1656 = buf1626; del buf1626  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1655, (128, 512), (512, 1), 0), permute_1894, out=buf1656)
        del permute_1894
        buf1644 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1649 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1650 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1654 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1659 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1660 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_55, attention_output_16, mul_29], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf1641, buf1646, addmm_55, buf7, primals_57, primals_58, buf1651, buf1656, buf1644, buf1649, buf1650, buf1654, buf1659, buf1660, 128, 128, grid=grid(128), stream=stream0)
        del addmm_55
        del buf1641
        del primals_58
        buf1647 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1645, (512, 128), (1, 512), 0), view_152, out=buf1647)
        del view_152
        buf1648 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1645, buf1648, 512, 128, grid=grid(512), stream=stream0)
        buf1653 = reinterpret_tensor(buf1645, (128, 512), (512, 1), 0); del buf1645  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1651, (128, 128), (1, 128), 0), view_150, out=buf1653)
        del view_150
        buf1657 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1655, (512, 128), (1, 512), 0), view_148, out=buf1657)
        del view_148
        buf1658 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1655, buf1658, 512, 128, grid=grid(512), stream=stream0)
        buf1661 = buf1651; del buf1651  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1661, buf1656, primals_57, 16384, grid=grid(16384), stream=stream0)
        del primals_57
        buf1662 = reinterpret_tensor(buf1655, (128, 512), (512, 1), 0); del buf1655  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1661, (128, 128), (128, 1), 0), permute_1898, out=buf1662)
        del permute_1898
        buf1663 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1661, (128, 128), (1, 128), 0), view_146, out=buf1663)
        del view_146
        buf1665 = reinterpret_tensor(buf1662, (1, 128, 512), (65536, 512, 1), 0); del buf1662  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1665, le_84, 65536, grid=grid(65536), stream=stream0)
        del le_84
        buf1666 = buf1656; del buf1656  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1665, (128, 512), (512, 1), 0), permute_1902, out=buf1666)
        del permute_1902
        buf1671 = buf7; del buf7  # reuse
        buf1696 = reinterpret_tensor(buf1646, (1, 128, 128), (16384, 128, 1), 0); del buf1646  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf1661, buf1666, primals_55, primals_51, buf1671, buf1696, 16384, grid=grid(16384), stream=stream0)
        del primals_55
        buf1664 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1669 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1670 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1674 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1694 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1695 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_51, layer_input_19, mul_26], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf1661, buf1666, addmm_51, addmm_46, primals_51, primals_52, buf1671, buf1664, buf1669, buf1670, buf1674, buf1694, buf1695, 128, 128, grid=grid(128), stream=stream0)
        del addmm_46
        del addmm_51
        del primals_51
        del primals_52
        buf1667 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1665, (512, 128), (1, 512), 0), view_144, out=buf1667)
        del view_144
        buf1668 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1665, buf1668, 512, 128, grid=grid(512), stream=stream0)
        buf1672 = buf1666; del buf1666  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1671, (128, 128), (128, 1), 0), permute_1906, out=buf1672)
        del permute_1906
        buf1673 = reinterpret_tensor(buf1661, (128, 128), (128, 1), 0); del buf1661  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1671, (128, 128), (1, 128), 0), view_142, out=buf1673)
        del view_142
        # Source Nodes: [], Original ATen: []
        buf1675 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1672, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_60, clone_default_61, clone_default_62, None, alias_default_41, getitem_191, getitem_192, getitem_193, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_41
        del clone_default_60
        del clone_default_61
        del clone_default_62
        del getitem_191
        del getitem_192
        del getitem_193
        buf1676 = buf1675[0]
        buf1677 = buf1675[1]
        buf1678 = buf1675[2]
        del buf1675
        buf1679 = reinterpret_tensor(buf1665, (128, 512), (512, 1), 0); del buf1665  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1678, (128, 128), (128, 1), 0), permute_1919, out=buf1679)
        del permute_1919
        buf1680 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1678, (128, 128), (1, 128), 0), view_122, out=buf1680)
        buf1681 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1678, buf1681, 128, 128, grid=grid(128), stream=stream0)
        buf1682 = reinterpret_tensor(buf1678, (128, 128), (128, 1), 0); del buf1678  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1677, (128, 128), (128, 1), 0), permute_1923, out=buf1682)
        del permute_1923
        buf1683 = buf1672; del buf1672  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1677, (128, 128), (1, 128), 0), view_126, out=buf1683)
        buf1684 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1677, buf1684, 128, 128, grid=grid(128), stream=stream0)
        buf1685 = reinterpret_tensor(buf1677, (128, 128), (128, 1), 0); del buf1677  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1676, (128, 128), (128, 1), 0), permute_1927, out=buf1685)
        del permute_1927
        buf1686 = reinterpret_tensor(buf1671, (128, 128), (128, 1), 0); del buf1671  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1676, (128, 128), (1, 128), 0), view_126, out=buf1686)
        del view_126
        buf1687 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1676, buf1687, 128, 128, grid=grid(128), stream=stream0)
        del buf1676
        buf1688 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1689 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf1682, buf1685, addmm_47, buf1688, buf1689, 128, 128, grid=grid(128), stream=stream0)
        del addmm_47
        buf1690 = reinterpret_tensor(buf1682, (1, 128, 128), (16384, 128, 1), 0); del buf1682  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1690, buf1685, primals_53, 16384, grid=grid(16384), stream=stream0)
        del primals_53
        buf1691 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1690, (128, 128), (128, 1), 0), permute_1931, out=buf1691)
        del permute_1931
        buf1692 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1690, (128, 128), (1, 128), 0), view_122, out=buf1692)
        buf1693 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1690, buf1693, 128, 128, grid=grid(128), stream=stream0)
        buf1697 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1696, (128, 128), (128, 1), 0), permute_1935, out=buf1697)
        del permute_1935
        buf1698 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1696, (128, 128), (1, 128), 0), view_122, out=buf1698)
        del view_122
        buf1699 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1696, buf1699, 128, 128, grid=grid(128), stream=stream0)
        buf1700 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1702 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_26.run(buf1624, buf1679, buf1691, buf1697, buf6, buf1700, buf1702, 512, 128, grid=grid(512), stream=stream0)
        buf1701 = buf1624; del buf1624  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf1701, buf1679, buf1691, buf1697, primals_49, 65536, grid=grid(65536), stream=stream0)
        del primals_49
        buf1703 = reinterpret_tensor(buf1696, (128, 128), (128, 1), 0); del buf1696  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1701, (128, 512), (512, 1), 0), permute_1939, out=buf1703)
        del permute_1939
        buf1704 = reinterpret_tensor(buf1697, (512, 128), (128, 1), 0); del buf1697  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1701, (512, 128), (1, 512), 0), view_120, out=buf1704)
        del view_120
        buf1705 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf1701, buf1705, 512, 128, grid=grid(512), stream=stream0)
        buf1708 = buf1690; del buf1690  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf1703, primals_47, buf1708, 16384, grid=grid(16384), stream=stream0)
        del primals_47
        buf1709 = buf1691; del buf1691  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1708, (128, 128), (128, 1), 0), permute_1943, out=buf1709)
        del permute_1943
        buf1712 = reinterpret_tensor(buf1709, (1, 128, 512), (65536, 512, 1), 0); del buf1709  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1712, le_85, 65536, grid=grid(65536), stream=stream0)
        del le_85
        buf1713 = buf1685; del buf1685  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1712, (128, 512), (512, 1), 0), permute_1947, out=buf1713)
        del permute_1947
        buf1706 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1707 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1711 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1716 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1717 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_44, attention_output_13, mul_23], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf1703, addmm_44, buf5, primals_45, primals_46, buf1708, buf1713, buf1706, buf1707, buf1711, buf1716, buf1717, 128, 128, grid=grid(128), stream=stream0)
        del addmm_44
        del primals_46
        buf1710 = buf1679; del buf1679  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1708, (128, 128), (1, 128), 0), view_118, out=buf1710)
        del view_118
        buf1714 = reinterpret_tensor(buf6, (512, 128), (128, 1), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1712, (512, 128), (1, 512), 0), view_116, out=buf1714)
        del view_116
        buf1715 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1712, buf1715, 512, 128, grid=grid(512), stream=stream0)
        buf1718 = buf1708; del buf1708  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1718, buf1713, primals_45, 16384, grid=grid(16384), stream=stream0)
        del primals_45
        buf1719 = reinterpret_tensor(buf1712, (128, 512), (512, 1), 0); del buf1712  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1718, (128, 128), (128, 1), 0), permute_1951, out=buf1719)
        del permute_1951
        buf1720 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1718, (128, 128), (1, 128), 0), view_114, out=buf1720)
        del view_114
        buf1722 = reinterpret_tensor(buf1719, (1, 128, 512), (65536, 512, 1), 0); del buf1719  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1722, le_86, 65536, grid=grid(65536), stream=stream0)
        del le_86
        buf1723 = buf1713; del buf1713  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1722, (128, 512), (512, 1), 0), permute_1955, out=buf1723)
        del permute_1955
        buf1728 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf1718, buf1723, primals_43, buf1728, 16384, grid=grid(16384), stream=stream0)
        del primals_43
        buf1729 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1728, (128, 128), (128, 1), 0), permute_1959, out=buf1729)
        del permute_1959
        buf1732 = reinterpret_tensor(buf1729, (1, 128, 512), (65536, 512, 1), 0); del buf1729  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1732, le_87, 65536, grid=grid(65536), stream=stream0)
        del le_87
        buf1733 = buf1703; del buf1703  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1732, (128, 512), (512, 1), 0), permute_1963, out=buf1733)
        del permute_1963
        buf1721 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1726 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1727 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1731 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1736 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1737 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_40, attention_output_11, mul_21], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf1718, buf1723, addmm_40, buf4, primals_41, primals_42, buf1728, buf1733, buf1721, buf1726, buf1727, buf1731, buf1736, buf1737, 128, 128, grid=grid(128), stream=stream0)
        del addmm_40
        del buf1718
        del primals_42
        buf1724 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1722, (512, 128), (1, 512), 0), view_112, out=buf1724)
        del view_112
        buf1725 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1722, buf1725, 512, 128, grid=grid(512), stream=stream0)
        buf1730 = reinterpret_tensor(buf1722, (128, 512), (512, 1), 0); del buf1722  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1728, (128, 128), (1, 128), 0), view_110, out=buf1730)
        del view_110
        buf1734 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1732, (512, 128), (1, 512), 0), view_108, out=buf1734)
        del view_108
        buf1735 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1732, buf1735, 512, 128, grid=grid(512), stream=stream0)
        buf1738 = buf1728; del buf1728  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1738, buf1733, primals_41, 16384, grid=grid(16384), stream=stream0)
        del primals_41
        buf1739 = reinterpret_tensor(buf1732, (128, 512), (512, 1), 0); del buf1732  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1738, (128, 128), (128, 1), 0), permute_1967, out=buf1739)
        del permute_1967
        buf1740 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1738, (128, 128), (1, 128), 0), view_106, out=buf1740)
        del view_106
        buf1742 = reinterpret_tensor(buf1739, (1, 128, 512), (65536, 512, 1), 0); del buf1739  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1742, le_88, 65536, grid=grid(65536), stream=stream0)
        del le_88
        buf1743 = buf1733; del buf1733  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1742, (128, 512), (512, 1), 0), permute_1971, out=buf1743)
        del permute_1971
        buf1748 = buf4; del buf4  # reuse
        buf1773 = reinterpret_tensor(buf1723, (1, 128, 128), (16384, 128, 1), 0); del buf1723  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf1738, buf1743, primals_39, primals_35, buf1748, buf1773, 16384, grid=grid(16384), stream=stream0)
        del primals_39
        buf1741 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1746 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1747 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1751 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1771 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1772 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_36, layer_input_14, mul_18], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf1738, buf1743, addmm_36, addmm_31, primals_35, primals_36, buf1748, buf1741, buf1746, buf1747, buf1751, buf1771, buf1772, 128, 128, grid=grid(128), stream=stream0)
        del addmm_31
        del addmm_36
        del primals_35
        del primals_36
        buf1744 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1742, (512, 128), (1, 512), 0), view_104, out=buf1744)
        del view_104
        buf1745 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1742, buf1745, 512, 128, grid=grid(512), stream=stream0)
        buf1749 = buf1743; del buf1743  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1748, (128, 128), (128, 1), 0), permute_1975, out=buf1749)
        del permute_1975
        buf1750 = reinterpret_tensor(buf1738, (128, 128), (128, 1), 0); del buf1738  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1748, (128, 128), (1, 128), 0), view_102, out=buf1750)
        del view_102
        # Source Nodes: [], Original ATen: []
        buf1752 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1749, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_63, clone_default_64, clone_default_65, None, alias_default_43, getitem_198, getitem_199, getitem_200, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_43
        del clone_default_63
        del clone_default_64
        del clone_default_65
        del getitem_198
        del getitem_199
        del getitem_200
        buf1753 = buf1752[0]
        buf1754 = buf1752[1]
        buf1755 = buf1752[2]
        del buf1752
        buf1756 = reinterpret_tensor(buf1742, (128, 512), (512, 1), 0); del buf1742  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1755, (128, 128), (128, 1), 0), permute_1988, out=buf1756)
        del permute_1988
        buf1757 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1755, (128, 128), (1, 128), 0), view_82, out=buf1757)
        buf1758 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1755, buf1758, 128, 128, grid=grid(128), stream=stream0)
        buf1759 = reinterpret_tensor(buf1755, (128, 128), (128, 1), 0); del buf1755  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1754, (128, 128), (128, 1), 0), permute_1992, out=buf1759)
        del permute_1992
        buf1760 = buf1749; del buf1749  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1754, (128, 128), (1, 128), 0), view_86, out=buf1760)
        buf1761 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1754, buf1761, 128, 128, grid=grid(128), stream=stream0)
        buf1762 = reinterpret_tensor(buf1754, (128, 128), (128, 1), 0); del buf1754  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1753, (128, 128), (128, 1), 0), permute_1996, out=buf1762)
        del permute_1996
        buf1763 = reinterpret_tensor(buf1748, (128, 128), (128, 1), 0); del buf1748  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1753, (128, 128), (1, 128), 0), view_86, out=buf1763)
        del view_86
        buf1764 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1753, buf1764, 128, 128, grid=grid(128), stream=stream0)
        del buf1753
        buf1765 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1766 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf1759, buf1762, addmm_32, buf1765, buf1766, 128, 128, grid=grid(128), stream=stream0)
        del addmm_32
        buf1767 = reinterpret_tensor(buf1759, (1, 128, 128), (16384, 128, 1), 0); del buf1759  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1767, buf1762, primals_37, 16384, grid=grid(16384), stream=stream0)
        del primals_37
        buf1768 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1767, (128, 128), (128, 1), 0), permute_2000, out=buf1768)
        del permute_2000
        buf1769 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1767, (128, 128), (1, 128), 0), view_82, out=buf1769)
        buf1770 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1767, buf1770, 128, 128, grid=grid(128), stream=stream0)
        buf1774 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1773, (128, 128), (128, 1), 0), permute_2004, out=buf1774)
        del permute_2004
        buf1775 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1773, (128, 128), (1, 128), 0), view_82, out=buf1775)
        del view_82
        buf1776 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1773, buf1776, 128, 128, grid=grid(128), stream=stream0)
        buf1777 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1779 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_31, mul_9, value_tensor_1], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_28.run(buf1701, buf1756, buf1768, buf1774, addmm_30, add_16, primals_17, primals_18, buf1777, buf1779, 512, 128, grid=grid(512), stream=stream0)
        del addmm_30
        del primals_18
        buf1778 = buf1701; del buf1701  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf1778, buf1756, buf1768, buf1774, primals_33, 65536, grid=grid(65536), stream=stream0)
        del primals_33
        buf1780 = reinterpret_tensor(buf1773, (128, 128), (128, 1), 0); del buf1773  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1778, (128, 512), (512, 1), 0), permute_2008, out=buf1780)
        del permute_2008
        buf1781 = reinterpret_tensor(buf1774, (512, 128), (128, 1), 0); del buf1774  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1778, (512, 128), (1, 512), 0), view_80, out=buf1781)
        del view_80
        buf1782 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf1778, buf1782, 512, 128, grid=grid(512), stream=stream0)
        buf1785 = buf1767; del buf1767  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf1780, primals_31, buf1785, 16384, grid=grid(16384), stream=stream0)
        del primals_31
        buf1786 = buf1768; del buf1768  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1785, (128, 128), (128, 1), 0), permute_2012, out=buf1786)
        del permute_2012
        buf1789 = reinterpret_tensor(buf1786, (1, 128, 512), (65536, 512, 1), 0); del buf1786  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1789, le_89, 65536, grid=grid(65536), stream=stream0)
        del le_89
        buf1790 = buf1762; del buf1762  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1789, (128, 512), (512, 1), 0), permute_2016, out=buf1790)
        del permute_2016
        buf1783 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1784 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1788 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1793 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1794 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_29, attention_output_8, mul_15], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf1780, addmm_29, buf3, primals_29, primals_30, buf1785, buf1790, buf1783, buf1784, buf1788, buf1793, buf1794, 128, 128, grid=grid(128), stream=stream0)
        del addmm_29
        del primals_30
        buf1787 = buf1756; del buf1756  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1785, (128, 128), (1, 128), 0), view_78, out=buf1787)
        del view_78
        buf1791 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1789, (512, 128), (1, 512), 0), view_76, out=buf1791)
        del view_76
        buf1792 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1789, buf1792, 512, 128, grid=grid(512), stream=stream0)
        buf1795 = buf1785; del buf1785  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1795, buf1790, primals_29, 16384, grid=grid(16384), stream=stream0)
        del primals_29
        buf1796 = reinterpret_tensor(buf1789, (128, 512), (512, 1), 0); del buf1789  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1795, (128, 128), (128, 1), 0), permute_2020, out=buf1796)
        del permute_2020
        buf1797 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1795, (128, 128), (1, 128), 0), view_74, out=buf1797)
        del view_74
        buf1799 = reinterpret_tensor(buf1796, (1, 128, 512), (65536, 512, 1), 0); del buf1796  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1799, le_90, 65536, grid=grid(65536), stream=stream0)
        del le_90
        buf1800 = buf1790; del buf1790  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1799, (128, 512), (512, 1), 0), permute_2024, out=buf1800)
        del permute_2024
        buf1805 = buf3; del buf3  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf1795, buf1800, primals_27, buf1805, 16384, grid=grid(16384), stream=stream0)
        del primals_27
        buf1806 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1805, (128, 128), (128, 1), 0), permute_2028, out=buf1806)
        del permute_2028
        buf1809 = reinterpret_tensor(buf1806, (1, 128, 512), (65536, 512, 1), 0); del buf1806  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1809, le_91, 65536, grid=grid(65536), stream=stream0)
        del le_91
        buf1810 = buf1780; del buf1780  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1809, (128, 512), (512, 1), 0), permute_2032, out=buf1810)
        del permute_2032
        buf1798 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1803 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1804 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1808 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1813 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1814 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_25, attention_output_6, mul_13], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf1795, buf1800, addmm_25, buf2, primals_25, primals_26, buf1805, buf1810, buf1798, buf1803, buf1804, buf1808, buf1813, buf1814, 128, 128, grid=grid(128), stream=stream0)
        del addmm_25
        del buf1795
        del primals_26
        buf1801 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1799, (512, 128), (1, 512), 0), view_72, out=buf1801)
        del view_72
        buf1802 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1799, buf1802, 512, 128, grid=grid(512), stream=stream0)
        buf1807 = reinterpret_tensor(buf1799, (128, 512), (512, 1), 0); del buf1799  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1805, (128, 128), (1, 128), 0), view_70, out=buf1807)
        del view_70
        buf1811 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1809, (512, 128), (1, 512), 0), view_68, out=buf1811)
        del view_68
        buf1812 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1809, buf1812, 512, 128, grid=grid(512), stream=stream0)
        buf1815 = buf1805; del buf1805  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1815, buf1810, primals_25, 16384, grid=grid(16384), stream=stream0)
        del primals_25
        buf1816 = reinterpret_tensor(buf1809, (128, 512), (512, 1), 0); del buf1809  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1815, (128, 128), (128, 1), 0), permute_2036, out=buf1816)
        del permute_2036
        buf1817 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1815, (128, 128), (1, 128), 0), view_66, out=buf1817)
        del view_66
        buf1819 = reinterpret_tensor(buf1816, (1, 128, 512), (65536, 512, 1), 0); del buf1816  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1819, le_92, 65536, grid=grid(65536), stream=stream0)
        del le_92
        buf1820 = buf1810; del buf1810  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1819, (128, 512), (512, 1), 0), permute_2040, out=buf1820)
        del permute_2040
        buf1825 = buf2; del buf2  # reuse
        buf1850 = reinterpret_tensor(buf1800, (1, 128, 128), (16384, 128, 1), 0); del buf1800  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf1815, buf1820, primals_23, primals_19, buf1825, buf1850, 16384, grid=grid(16384), stream=stream0)
        del primals_23
        buf1818 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1823 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1824 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1828 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1848 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1849 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_21, layer_input_9, mul_10], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf1815, buf1820, addmm_21, addmm_16, primals_19, primals_20, buf1825, buf1818, buf1823, buf1824, buf1828, buf1848, buf1849, 128, 128, grid=grid(128), stream=stream0)
        del addmm_16
        del addmm_21
        del primals_19
        del primals_20
        buf1821 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1819, (512, 128), (1, 512), 0), view_64, out=buf1821)
        del view_64
        buf1822 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1819, buf1822, 512, 128, grid=grid(512), stream=stream0)
        buf1826 = buf1820; del buf1820  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1825, (128, 128), (128, 1), 0), permute_2044, out=buf1826)
        del permute_2044
        buf1827 = reinterpret_tensor(buf1815, (128, 128), (128, 1), 0); del buf1815  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1825, (128, 128), (1, 128), 0), view_62, out=buf1827)
        del view_62
        # Source Nodes: [], Original ATen: []
        buf1829 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1826, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_66, clone_default_67, clone_default_68, None, alias_default_45, getitem_205, getitem_206, getitem_207, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_45
        del clone_default_66
        del clone_default_67
        del clone_default_68
        del getitem_205
        del getitem_206
        del getitem_207
        buf1830 = buf1829[0]
        buf1831 = buf1829[1]
        buf1832 = buf1829[2]
        del buf1829
        buf1833 = reinterpret_tensor(buf1819, (128, 512), (512, 1), 0); del buf1819  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1832, (128, 128), (128, 1), 0), permute_2057, out=buf1833)
        del permute_2057
        buf1834 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1832, (128, 128), (1, 128), 0), view_42, out=buf1834)
        buf1835 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1832, buf1835, 128, 128, grid=grid(128), stream=stream0)
        buf1836 = reinterpret_tensor(buf1832, (128, 128), (128, 1), 0); del buf1832  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1831, (128, 128), (128, 1), 0), permute_2061, out=buf1836)
        del permute_2061
        buf1837 = buf1826; del buf1826  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1831, (128, 128), (1, 128), 0), view_46, out=buf1837)
        buf1838 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1831, buf1838, 128, 128, grid=grid(128), stream=stream0)
        buf1839 = reinterpret_tensor(buf1831, (128, 128), (128, 1), 0); del buf1831  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1830, (128, 128), (128, 1), 0), permute_2065, out=buf1839)
        del permute_2065
        buf1840 = reinterpret_tensor(buf1825, (128, 128), (128, 1), 0); del buf1825  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1830, (128, 128), (1, 128), 0), view_46, out=buf1840)
        del view_46
        buf1841 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1830, buf1841, 128, 128, grid=grid(128), stream=stream0)
        del buf1830
        buf1842 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1843 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf1836, buf1839, addmm_17, buf1842, buf1843, 128, 128, grid=grid(128), stream=stream0)
        del addmm_17
        buf1844 = reinterpret_tensor(buf1836, (1, 128, 128), (16384, 128, 1), 0); del buf1836  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1844, buf1839, primals_21, 16384, grid=grid(16384), stream=stream0)
        del primals_21
        buf1845 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1844, (128, 128), (128, 1), 0), permute_2069, out=buf1845)
        del permute_2069
        buf1846 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1844, (128, 128), (1, 128), 0), view_42, out=buf1846)
        buf1847 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1844, buf1847, 128, 128, grid=grid(128), stream=stream0)
        buf1851 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1850, (128, 128), (128, 1), 0), permute_2073, out=buf1851)
        del permute_2073
        buf1852 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1850, (128, 128), (1, 128), 0), view_42, out=buf1852)
        del view_42
        buf1853 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1850, buf1853, 128, 128, grid=grid(128), stream=stream0)
        buf1854 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1856 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_26.run(buf1778, buf1833, buf1845, buf1851, add_16, buf1854, buf1856, 512, 128, grid=grid(512), stream=stream0)
        del add_16
        buf1855 = buf1778; del buf1778  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_29.run(buf1855, buf1833, buf1845, buf1851, primals_17, 65536, grid=grid(65536), stream=stream0)
        del primals_17
        buf1857 = reinterpret_tensor(buf1850, (128, 128), (128, 1), 0); del buf1850  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1855, (128, 512), (512, 1), 0), permute_2077, out=buf1857)
        del permute_2077
        buf1858 = reinterpret_tensor(buf1851, (512, 128), (128, 1), 0); del buf1851  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1855, (512, 128), (1, 512), 0), view_40, out=buf1858)
        del view_40
        buf1859 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_18.run(buf1855, buf1859, 512, 128, grid=grid(512), stream=stream0)
        buf1862 = buf1844; del buf1844  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf1857, primals_15, buf1862, 16384, grid=grid(16384), stream=stream0)
        del primals_15
        buf1863 = buf1845; del buf1845  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1862, (128, 128), (128, 1), 0), permute_2081, out=buf1863)
        del permute_2081
        buf1866 = reinterpret_tensor(buf1863, (1, 128, 512), (65536, 512, 1), 0); del buf1863  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1866, le_93, 65536, grid=grid(65536), stream=stream0)
        del le_93
        buf1867 = buf1839; del buf1839  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1866, (128, 512), (512, 1), 0), permute_2085, out=buf1867)
        del permute_2085
        buf1860 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1861 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1865 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1870 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1871 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_14, attention_output_3, mul_7], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_19.run(buf1857, addmm_14, buf1, primals_13, primals_14, buf1862, buf1867, buf1860, buf1861, buf1865, buf1870, buf1871, 128, 128, grid=grid(128), stream=stream0)
        del addmm_14
        del primals_14
        buf1864 = buf1833; del buf1833  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1862, (128, 128), (1, 128), 0), view_38, out=buf1864)
        del view_38
        buf1868 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1866, (512, 128), (1, 512), 0), view_36, out=buf1868)
        del view_36
        buf1869 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1866, buf1869, 512, 128, grid=grid(512), stream=stream0)
        buf1872 = buf1862; del buf1862  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1872, buf1867, primals_13, 16384, grid=grid(16384), stream=stream0)
        del primals_13
        buf1873 = reinterpret_tensor(buf1866, (128, 512), (512, 1), 0); del buf1866  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1872, (128, 128), (128, 1), 0), permute_2089, out=buf1873)
        del permute_2089
        buf1874 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1872, (128, 128), (1, 128), 0), view_34, out=buf1874)
        del view_34
        buf1876 = reinterpret_tensor(buf1873, (1, 128, 512), (65536, 512, 1), 0); del buf1873  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1876, le_94, 65536, grid=grid(65536), stream=stream0)
        del le_94
        buf1877 = buf1867; del buf1867  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1876, (128, 512), (512, 1), 0), permute_2093, out=buf1877)
        del permute_2093
        buf1882 = reinterpret_tensor(buf1857, (1, 128, 128), (16384, 128, 1), 0); del buf1857  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_12.run(buf1872, buf1877, primals_11, buf1882, 16384, grid=grid(16384), stream=stream0)
        del primals_11
        buf1883 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1882, (128, 128), (128, 1), 0), permute_2097, out=buf1883)
        del permute_2097
        buf1886 = reinterpret_tensor(buf1883, (1, 128, 512), (65536, 512, 1), 0); del buf1883  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1886, le_95, 65536, grid=grid(65536), stream=stream0)
        del le_95
        buf1887 = reinterpret_tensor(buf1, (128, 128), (128, 1), 0); del buf1  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1886, (128, 512), (512, 1), 0), permute_2101, out=buf1887)
        del permute_2101
        buf1875 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1880 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1881 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1885 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1890 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1891 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_10, attention_output_1, mul_5], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_20.run(buf1872, buf1877, addmm_10, buf0, primals_9, primals_10, buf1882, buf1887, buf1875, buf1880, buf1881, buf1885, buf1890, buf1891, 128, 128, grid=grid(128), stream=stream0)
        del addmm_10
        del buf0
        del primals_10
        buf1878 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1876, (512, 128), (1, 512), 0), view_32, out=buf1878)
        del view_32
        buf1879 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1876, buf1879, 512, 128, grid=grid(512), stream=stream0)
        buf1884 = reinterpret_tensor(buf1876, (128, 512), (512, 1), 0); del buf1876  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1882, (128, 128), (1, 128), 0), view_30, out=buf1884)
        del view_30
        buf1888 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1886, (512, 128), (1, 512), 0), view_28, out=buf1888)
        del view_28
        buf1889 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1886, buf1889, 512, 128, grid=grid(512), stream=stream0)
        buf1892 = buf1882; del buf1882  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1892, buf1887, primals_9, 16384, grid=grid(16384), stream=stream0)
        del primals_9
        buf1893 = reinterpret_tensor(buf1886, (128, 512), (512, 1), 0); del buf1886  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1892, (128, 128), (128, 1), 0), permute_2105, out=buf1893)
        del permute_2105
        buf1894 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1892, (128, 128), (1, 128), 0), view_26, out=buf1894)
        del view_26
        buf1896 = reinterpret_tensor(buf1893, (1, 128, 512), (65536, 512, 1), 0); del buf1893  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_11.run(buf1896, le_96, 65536, grid=grid(65536), stream=stream0)
        del le_96
        buf1897 = buf1887; del buf1887  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1896, (128, 512), (512, 1), 0), permute_2109, out=buf1897)
        del permute_2109
        buf1902 = reinterpret_tensor(buf1877, (1, 128, 128), (16384, 128, 1), 0); del buf1877  # reuse
        buf1927 = buf1872; del buf1872  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_22.run(buf1892, buf1897, primals_7, primals_3, buf1902, buf1927, 16384, grid=grid(16384), stream=stream0)
        del primals_7
        buf1895 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1900 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1901 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1905 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1925 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1926 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_6, layer_input_4, mul_2], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_23.run(buf1892, buf1897, addmm_6, addmm_1, primals_3, primals_4, buf1902, buf1895, buf1900, buf1901, buf1905, buf1925, buf1926, 128, 128, grid=grid(128), stream=stream0)
        del addmm_1
        del addmm_6
        del primals_3
        del primals_4
        buf1898 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1896, (512, 128), (1, 512), 0), view_24, out=buf1898)
        del view_24
        buf1899 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1896, buf1899, 512, 128, grid=grid(512), stream=stream0)
        buf1903 = buf1897; del buf1897  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1902, (128, 128), (128, 1), 0), permute_2113, out=buf1903)
        del permute_2113
        buf1904 = reinterpret_tensor(buf1892, (128, 128), (128, 1), 0); del buf1892  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1902, (128, 128), (1, 128), 0), view_22, out=buf1904)
        del view_22
        # Source Nodes: [], Original ATen: []
        buf1906 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1903, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_69, clone_default_70, clone_default_71, None, alias_default_47, getitem_212, getitem_213, getitem_214, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_47
        del clone_default_69
        del clone_default_70
        del clone_default_71
        del getitem_212
        del getitem_213
        del getitem_214
        buf1907 = buf1906[0]
        buf1908 = buf1906[1]
        buf1909 = buf1906[2]
        del buf1906
        buf1910 = reinterpret_tensor(buf1896, (128, 512), (512, 1), 0); del buf1896  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1909, (128, 128), (128, 1), 0), permute_2126, out=buf1910)
        del permute_2126
        buf1911 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1909, (128, 128), (1, 128), 0), view_2, out=buf1911)
        buf1912 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1909, buf1912, 128, 128, grid=grid(128), stream=stream0)
        buf1913 = reinterpret_tensor(buf1909, (128, 128), (128, 1), 0); del buf1909  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1908, (128, 128), (128, 1), 0), permute_2130, out=buf1913)
        del permute_2130
        buf1914 = buf1903; del buf1903  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1908, (128, 128), (1, 128), 0), view_6, out=buf1914)
        buf1915 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1908, buf1915, 128, 128, grid=grid(128), stream=stream0)
        buf1916 = reinterpret_tensor(buf1908, (128, 128), (128, 1), 0); del buf1908  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1907, (128, 128), (128, 1), 0), permute_2134, out=buf1916)
        del permute_2134
        buf1917 = reinterpret_tensor(buf1902, (128, 128), (128, 1), 0); del buf1902  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1907, (128, 128), (1, 128), 0), view_6, out=buf1917)
        del view_6
        buf1918 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1907, buf1918, 128, 128, grid=grid(128), stream=stream0)
        del buf1907
        buf1919 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1920 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_25.run(buf1913, buf1916, addmm_2, buf1919, buf1920, 128, 128, grid=grid(128), stream=stream0)
        del addmm_2
        buf1921 = reinterpret_tensor(buf1913, (1, 128, 128), (16384, 128, 1), 0); del buf1913  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1921, buf1916, primals_5, 16384, grid=grid(16384), stream=stream0)
        del buf1916
        del primals_5
        buf1922 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1921, (128, 128), (128, 1), 0), permute_2138, out=buf1922)
        del permute_2138
        buf1923 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1921, (128, 128), (1, 128), 0), view_2, out=buf1923)
        buf1924 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1921, buf1924, 128, 128, grid=grid(128), stream=stream0)
        del buf1921
        buf1928 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1927, (128, 128), (128, 1), 0), permute_2142, out=buf1928)
        del permute_2142
        buf1929 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1927, (128, 128), (1, 128), 0), view_2, out=buf1929)
        del view_2
        buf1930 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1927, buf1930, 128, 128, grid=grid(128), stream=stream0)
        buf1931 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1933 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_26.run(buf1855, buf1910, buf1922, buf1928, add_1, buf1931, buf1933, 512, 128, grid=grid(512), stream=stream0)
        del add_1
        buf1932 = buf1855; del buf1855  # reuse
        buf1935 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        buf1939 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.mul, aten.nll_loss_forward]
        triton_poi_fused_add_embedding_dense_backward_mul_nll_loss_forward_31.run(buf1932, buf1910, buf1922, buf1928, primals_1, slice_4, buf1935, buf1939, 65536, grid=grid(65536), stream=stream0)
        del buf1910
        del buf1922
        del buf1928
        del primals_1
        buf1934 = empty((2, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_32.run(buf1934, 1024, grid=grid(1024), stream=stream0)
        aten.index_put_(buf1934, [full_default], buf1935, True)
        del buf1935
        del full_default
        buf1938 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_33.run(buf1938, 262144, grid=grid(262144), stream=stream0)
        aten.index_put_(buf1938, [slice_4], buf1939, True)
        del buf1939
        del slice_4
        buf1942 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1932, (128, 512), (512, 1), 0), permute_2146, out=buf1942)
        del permute_2146
        buf1943 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1932, (512, 128), (1, 512), 0), view, out=buf1943)
        del view
        buf1944 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf1932, buf1944, 512, 128, grid=grid(512), stream=stream0)
        del buf1932
        buf1945 = reinterpret_tensor(buf70, (30522, 128), (128, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_2.run(buf1945, 3906816, grid=grid(3906816), stream=stream0)
        buf1946 = buf1927; del buf1927  # reuse
        # Source Nodes: [loss], Original ATen: [aten.add, aten.constant_pad_nd, aten.embedding_dense_backward, aten.nll_loss_forward, aten.slice_backward]
        triton_poi_fused_add_constant_pad_nd_embedding_dense_backward_nll_loss_forward_slice_backward_34.run(primals_1120, buf1942, buf1946, 16384, grid=grid(16384), stream=stream0)
        del buf1942
        aten.index_put_(buf1945, [primals_1120], buf1946, True)
        del buf1946
        del primals_1120
        return (reinterpret_tensor(buf1933, (512, ), (1, ), 0), reinterpret_tensor(buf1931, (512, ), (1, ), 0), reinterpret_tensor(buf1926, (128, ), (1, ), 0), reinterpret_tensor(buf1925, (128, ), (1, ), 0), reinterpret_tensor(buf1920, (128, ), (1, ), 0), reinterpret_tensor(buf1919, (128, ), (1, ), 0), reinterpret_tensor(buf1901, (128, ), (1, ), 0), reinterpret_tensor(buf1900, (128, ), (1, ), 0), reinterpret_tensor(buf1891, (128, ), (1, ), 0), reinterpret_tensor(buf1890, (128, ), (1, ), 0), reinterpret_tensor(buf1881, (128, ), (1, ), 0), reinterpret_tensor(buf1880, (128, ), (1, ), 0), reinterpret_tensor(buf1871, (128, ), (1, ), 0), reinterpret_tensor(buf1870, (128, ), (1, ), 0), reinterpret_tensor(buf1861, (128, ), (1, ), 0), reinterpret_tensor(buf1860, (128, ), (1, ), 0), reinterpret_tensor(buf1856, (512, ), (1, ), 0), reinterpret_tensor(buf1854, (512, ), (1, ), 0), reinterpret_tensor(buf1849, (128, ), (1, ), 0), reinterpret_tensor(buf1848, (128, ), (1, ), 0), reinterpret_tensor(buf1843, (128, ), (1, ), 0), reinterpret_tensor(buf1842, (128, ), (1, ), 0), reinterpret_tensor(buf1824, (128, ), (1, ), 0), reinterpret_tensor(buf1823, (128, ), (1, ), 0), reinterpret_tensor(buf1814, (128, ), (1, ), 0), reinterpret_tensor(buf1813, (128, ), (1, ), 0), reinterpret_tensor(buf1804, (128, ), (1, ), 0), reinterpret_tensor(buf1803, (128, ), (1, ), 0), reinterpret_tensor(buf1794, (128, ), (1, ), 0), reinterpret_tensor(buf1793, (128, ), (1, ), 0), reinterpret_tensor(buf1784, (128, ), (1, ), 0), reinterpret_tensor(buf1783, (128, ), (1, ), 0), reinterpret_tensor(buf1779, (512, ), (1, ), 0), reinterpret_tensor(buf1777, (512, ), (1, ), 0), reinterpret_tensor(buf1772, (128, ), (1, ), 0), reinterpret_tensor(buf1771, (128, ), (1, ), 0), reinterpret_tensor(buf1766, (128, ), (1, ), 0), reinterpret_tensor(buf1765, (128, ), (1, ), 0), reinterpret_tensor(buf1747, (128, ), (1, ), 0), reinterpret_tensor(buf1746, (128, ), (1, ), 0), reinterpret_tensor(buf1737, (128, ), (1, ), 0), reinterpret_tensor(buf1736, (128, ), (1, ), 0), reinterpret_tensor(buf1727, (128, ), (1, ), 0), reinterpret_tensor(buf1726, (128, ), (1, ), 0), reinterpret_tensor(buf1717, (128, ), (1, ), 0), reinterpret_tensor(buf1716, (128, ), (1, ), 0), reinterpret_tensor(buf1707, (128, ), (1, ), 0), reinterpret_tensor(buf1706, (128, ), (1, ), 0), reinterpret_tensor(buf1702, (512, ), (1, ), 0), reinterpret_tensor(buf1700, (512, ), (1, ), 0), reinterpret_tensor(buf1695, (128, ), (1, ), 0), reinterpret_tensor(buf1694, (128, ), (1, ), 0), reinterpret_tensor(buf1689, (128, ), (1, ), 0), reinterpret_tensor(buf1688, (128, ), (1, ), 0), reinterpret_tensor(buf1670, (128, ), (1, ), 0), reinterpret_tensor(buf1669, (128, ), (1, ), 0), reinterpret_tensor(buf1660, (128, ), (1, ), 0), reinterpret_tensor(buf1659, (128, ), (1, ), 0), reinterpret_tensor(buf1650, (128, ), (1, ), 0), reinterpret_tensor(buf1649, (128, ), (1, ), 0), reinterpret_tensor(buf1640, (128, ), (1, ), 0), reinterpret_tensor(buf1639, (128, ), (1, ), 0), reinterpret_tensor(buf1630, (128, ), (1, ), 0), reinterpret_tensor(buf1629, (128, ), (1, ), 0), reinterpret_tensor(buf1625, (512, ), (1, ), 0), reinterpret_tensor(buf1623, (512, ), (1, ), 0), reinterpret_tensor(buf1618, (128, ), (1, ), 0), reinterpret_tensor(buf1617, (128, ), (1, ), 0), reinterpret_tensor(buf1612, (128, ), (1, ), 0), reinterpret_tensor(buf1611, (128, ), (1, ), 0), reinterpret_tensor(buf1593, (128, ), (1, ), 0), reinterpret_tensor(buf1592, (128, ), (1, ), 0), reinterpret_tensor(buf1583, (128, ), (1, ), 0), reinterpret_tensor(buf1582, (128, ), (1, ), 0), reinterpret_tensor(buf1573, (128, ), (1, ), 0), reinterpret_tensor(buf1572, (128, ), (1, ), 0), reinterpret_tensor(buf1563, (128, ), (1, ), 0), reinterpret_tensor(buf1562, (128, ), (1, ), 0), reinterpret_tensor(buf1553, (128, ), (1, ), 0), reinterpret_tensor(buf1552, (128, ), (1, ), 0), reinterpret_tensor(buf1548, (512, ), (1, ), 0), reinterpret_tensor(buf1546, (512, ), (1, ), 0), reinterpret_tensor(buf1541, (128, ), (1, ), 0), reinterpret_tensor(buf1540, (128, ), (1, ), 0), reinterpret_tensor(buf1535, (128, ), (1, ), 0), reinterpret_tensor(buf1534, (128, ), (1, ), 0), reinterpret_tensor(buf1516, (128, ), (1, ), 0), reinterpret_tensor(buf1515, (128, ), (1, ), 0), reinterpret_tensor(buf1506, (128, ), (1, ), 0), reinterpret_tensor(buf1505, (128, ), (1, ), 0), reinterpret_tensor(buf1496, (128, ), (1, ), 0), reinterpret_tensor(buf1495, (128, ), (1, ), 0), reinterpret_tensor(buf1486, (128, ), (1, ), 0), reinterpret_tensor(buf1485, (128, ), (1, ), 0), reinterpret_tensor(buf1476, (128, ), (1, ), 0), reinterpret_tensor(buf1475, (128, ), (1, ), 0), reinterpret_tensor(buf1471, (512, ), (1, ), 0), reinterpret_tensor(buf1469, (512, ), (1, ), 0), reinterpret_tensor(buf1464, (128, ), (1, ), 0), reinterpret_tensor(buf1463, (128, ), (1, ), 0), reinterpret_tensor(buf1458, (128, ), (1, ), 0), reinterpret_tensor(buf1457, (128, ), (1, ), 0), reinterpret_tensor(buf1439, (128, ), (1, ), 0), reinterpret_tensor(buf1438, (128, ), (1, ), 0), reinterpret_tensor(buf1429, (128, ), (1, ), 0), reinterpret_tensor(buf1428, (128, ), (1, ), 0), reinterpret_tensor(buf1419, (128, ), (1, ), 0), reinterpret_tensor(buf1418, (128, ), (1, ), 0), reinterpret_tensor(buf1409, (128, ), (1, ), 0), reinterpret_tensor(buf1408, (128, ), (1, ), 0), reinterpret_tensor(buf1399, (128, ), (1, ), 0), reinterpret_tensor(buf1398, (128, ), (1, ), 0), reinterpret_tensor(buf1394, (512, ), (1, ), 0), reinterpret_tensor(buf1392, (512, ), (1, ), 0), reinterpret_tensor(buf1387, (128, ), (1, ), 0), reinterpret_tensor(buf1386, (128, ), (1, ), 0), reinterpret_tensor(buf1381, (128, ), (1, ), 0), reinterpret_tensor(buf1380, (128, ), (1, ), 0), reinterpret_tensor(buf1362, (128, ), (1, ), 0), reinterpret_tensor(buf1361, (128, ), (1, ), 0), reinterpret_tensor(buf1352, (128, ), (1, ), 0), reinterpret_tensor(buf1351, (128, ), (1, ), 0), reinterpret_tensor(buf1342, (128, ), (1, ), 0), reinterpret_tensor(buf1341, (128, ), (1, ), 0), reinterpret_tensor(buf1332, (128, ), (1, ), 0), reinterpret_tensor(buf1331, (128, ), (1, ), 0), reinterpret_tensor(buf1322, (128, ), (1, ), 0), reinterpret_tensor(buf1321, (128, ), (1, ), 0), reinterpret_tensor(buf1317, (512, ), (1, ), 0), reinterpret_tensor(buf1315, (512, ), (1, ), 0), reinterpret_tensor(buf1310, (128, ), (1, ), 0), reinterpret_tensor(buf1309, (128, ), (1, ), 0), reinterpret_tensor(buf1304, (128, ), (1, ), 0), reinterpret_tensor(buf1303, (128, ), (1, ), 0), reinterpret_tensor(buf1285, (128, ), (1, ), 0), reinterpret_tensor(buf1284, (128, ), (1, ), 0), reinterpret_tensor(buf1275, (128, ), (1, ), 0), reinterpret_tensor(buf1274, (128, ), (1, ), 0), reinterpret_tensor(buf1265, (128, ), (1, ), 0), reinterpret_tensor(buf1264, (128, ), (1, ), 0), reinterpret_tensor(buf1255, (128, ), (1, ), 0), reinterpret_tensor(buf1254, (128, ), (1, ), 0), reinterpret_tensor(buf1245, (128, ), (1, ), 0), reinterpret_tensor(buf1244, (128, ), (1, ), 0), reinterpret_tensor(buf1240, (512, ), (1, ), 0), reinterpret_tensor(buf1238, (512, ), (1, ), 0), reinterpret_tensor(buf1233, (128, ), (1, ), 0), reinterpret_tensor(buf1232, (128, ), (1, ), 0), reinterpret_tensor(buf1227, (128, ), (1, ), 0), reinterpret_tensor(buf1226, (128, ), (1, ), 0), reinterpret_tensor(buf1208, (128, ), (1, ), 0), reinterpret_tensor(buf1207, (128, ), (1, ), 0), reinterpret_tensor(buf1198, (128, ), (1, ), 0), reinterpret_tensor(buf1197, (128, ), (1, ), 0), reinterpret_tensor(buf1188, (128, ), (1, ), 0), reinterpret_tensor(buf1187, (128, ), (1, ), 0), reinterpret_tensor(buf1178, (128, ), (1, ), 0), reinterpret_tensor(buf1177, (128, ), (1, ), 0), reinterpret_tensor(buf1168, (128, ), (1, ), 0), reinterpret_tensor(buf1167, (128, ), (1, ), 0), reinterpret_tensor(buf1163, (512, ), (1, ), 0), reinterpret_tensor(buf1161, (512, ), (1, ), 0), reinterpret_tensor(buf1156, (128, ), (1, ), 0), reinterpret_tensor(buf1155, (128, ), (1, ), 0), reinterpret_tensor(buf1150, (128, ), (1, ), 0), reinterpret_tensor(buf1149, (128, ), (1, ), 0), reinterpret_tensor(buf1131, (128, ), (1, ), 0), reinterpret_tensor(buf1130, (128, ), (1, ), 0), reinterpret_tensor(buf1121, (128, ), (1, ), 0), reinterpret_tensor(buf1120, (128, ), (1, ), 0), reinterpret_tensor(buf1111, (128, ), (1, ), 0), reinterpret_tensor(buf1110, (128, ), (1, ), 0), reinterpret_tensor(buf1101, (128, ), (1, ), 0), reinterpret_tensor(buf1100, (128, ), (1, ), 0), reinterpret_tensor(buf1091, (128, ), (1, ), 0), reinterpret_tensor(buf1090, (128, ), (1, ), 0), reinterpret_tensor(buf1086, (512, ), (1, ), 0), reinterpret_tensor(buf1084, (512, ), (1, ), 0), reinterpret_tensor(buf1079, (128, ), (1, ), 0), reinterpret_tensor(buf1078, (128, ), (1, ), 0), reinterpret_tensor(buf1073, (128, ), (1, ), 0), reinterpret_tensor(buf1072, (128, ), (1, ), 0), reinterpret_tensor(buf1054, (128, ), (1, ), 0), reinterpret_tensor(buf1053, (128, ), (1, ), 0), reinterpret_tensor(buf1044, (128, ), (1, ), 0), reinterpret_tensor(buf1043, (128, ), (1, ), 0), reinterpret_tensor(buf1034, (128, ), (1, ), 0), reinterpret_tensor(buf1033, (128, ), (1, ), 0), reinterpret_tensor(buf1024, (128, ), (1, ), 0), reinterpret_tensor(buf1023, (128, ), (1, ), 0), reinterpret_tensor(buf1014, (128, ), (1, ), 0), reinterpret_tensor(buf1013, (128, ), (1, ), 0), reinterpret_tensor(buf1009, (512, ), (1, ), 0), reinterpret_tensor(buf1007, (512, ), (1, ), 0), reinterpret_tensor(buf1002, (128, ), (1, ), 0), reinterpret_tensor(buf1001, (128, ), (1, ), 0), reinterpret_tensor(buf996, (128, ), (1, ), 0), reinterpret_tensor(buf995, (128, ), (1, ), 0), reinterpret_tensor(buf977, (128, ), (1, ), 0), reinterpret_tensor(buf976, (128, ), (1, ), 0), reinterpret_tensor(buf967, (128, ), (1, ), 0), reinterpret_tensor(buf966, (128, ), (1, ), 0), reinterpret_tensor(buf957, (128, ), (1, ), 0), reinterpret_tensor(buf956, (128, ), (1, ), 0), reinterpret_tensor(buf947, (128, ), (1, ), 0), reinterpret_tensor(buf946, (128, ), (1, ), 0), reinterpret_tensor(buf937, (128, ), (1, ), 0), reinterpret_tensor(buf936, (128, ), (1, ), 0), reinterpret_tensor(buf932, (512, ), (1, ), 0), reinterpret_tensor(buf930, (512, ), (1, ), 0), reinterpret_tensor(buf925, (128, ), (1, ), 0), reinterpret_tensor(buf924, (128, ), (1, ), 0), reinterpret_tensor(buf919, (128, ), (1, ), 0), reinterpret_tensor(buf918, (128, ), (1, ), 0), reinterpret_tensor(buf900, (128, ), (1, ), 0), reinterpret_tensor(buf899, (128, ), (1, ), 0), reinterpret_tensor(buf890, (128, ), (1, ), 0), reinterpret_tensor(buf889, (128, ), (1, ), 0), reinterpret_tensor(buf880, (128, ), (1, ), 0), reinterpret_tensor(buf879, (128, ), (1, ), 0), reinterpret_tensor(buf870, (128, ), (1, ), 0), reinterpret_tensor(buf869, (128, ), (1, ), 0), reinterpret_tensor(buf860, (128, ), (1, ), 0), reinterpret_tensor(buf859, (128, ), (1, ), 0), reinterpret_tensor(buf855, (512, ), (1, ), 0), reinterpret_tensor(buf853, (512, ), (1, ), 0), reinterpret_tensor(buf848, (128, ), (1, ), 0), reinterpret_tensor(buf847, (128, ), (1, ), 0), reinterpret_tensor(buf842, (128, ), (1, ), 0), reinterpret_tensor(buf841, (128, ), (1, ), 0), reinterpret_tensor(buf823, (128, ), (1, ), 0), reinterpret_tensor(buf822, (128, ), (1, ), 0), reinterpret_tensor(buf813, (128, ), (1, ), 0), reinterpret_tensor(buf812, (128, ), (1, ), 0), reinterpret_tensor(buf803, (128, ), (1, ), 0), reinterpret_tensor(buf802, (128, ), (1, ), 0), reinterpret_tensor(buf793, (128, ), (1, ), 0), reinterpret_tensor(buf792, (128, ), (1, ), 0), reinterpret_tensor(buf783, (128, ), (1, ), 0), reinterpret_tensor(buf782, (128, ), (1, ), 0), reinterpret_tensor(buf778, (512, ), (1, ), 0), reinterpret_tensor(buf776, (512, ), (1, ), 0), reinterpret_tensor(buf771, (128, ), (1, ), 0), reinterpret_tensor(buf770, (128, ), (1, ), 0), reinterpret_tensor(buf765, (128, ), (1, ), 0), reinterpret_tensor(buf764, (128, ), (1, ), 0), reinterpret_tensor(buf746, (128, ), (1, ), 0), reinterpret_tensor(buf745, (128, ), (1, ), 0), reinterpret_tensor(buf736, (128, ), (1, ), 0), reinterpret_tensor(buf735, (128, ), (1, ), 0), reinterpret_tensor(buf726, (128, ), (1, ), 0), reinterpret_tensor(buf725, (128, ), (1, ), 0), reinterpret_tensor(buf716, (128, ), (1, ), 0), reinterpret_tensor(buf715, (128, ), (1, ), 0), reinterpret_tensor(buf706, (128, ), (1, ), 0), reinterpret_tensor(buf705, (128, ), (1, ), 0), reinterpret_tensor(buf701, (512, ), (1, ), 0), reinterpret_tensor(buf699, (512, ), (1, ), 0), reinterpret_tensor(buf694, (128, ), (1, ), 0), reinterpret_tensor(buf693, (128, ), (1, ), 0), reinterpret_tensor(buf688, (128, ), (1, ), 0), reinterpret_tensor(buf687, (128, ), (1, ), 0), reinterpret_tensor(buf669, (128, ), (1, ), 0), reinterpret_tensor(buf668, (128, ), (1, ), 0), reinterpret_tensor(buf659, (128, ), (1, ), 0), reinterpret_tensor(buf658, (128, ), (1, ), 0), reinterpret_tensor(buf649, (128, ), (1, ), 0), reinterpret_tensor(buf648, (128, ), (1, ), 0), reinterpret_tensor(buf639, (128, ), (1, ), 0), reinterpret_tensor(buf638, (128, ), (1, ), 0), reinterpret_tensor(buf629, (128, ), (1, ), 0), reinterpret_tensor(buf628, (128, ), (1, ), 0), reinterpret_tensor(buf624, (512, ), (1, ), 0), reinterpret_tensor(buf622, (512, ), (1, ), 0), reinterpret_tensor(buf617, (128, ), (1, ), 0), reinterpret_tensor(buf616, (128, ), (1, ), 0), reinterpret_tensor(buf611, (128, ), (1, ), 0), reinterpret_tensor(buf610, (128, ), (1, ), 0), reinterpret_tensor(buf592, (128, ), (1, ), 0), reinterpret_tensor(buf591, (128, ), (1, ), 0), reinterpret_tensor(buf582, (128, ), (1, ), 0), reinterpret_tensor(buf581, (128, ), (1, ), 0), reinterpret_tensor(buf572, (128, ), (1, ), 0), reinterpret_tensor(buf571, (128, ), (1, ), 0), reinterpret_tensor(buf562, (128, ), (1, ), 0), reinterpret_tensor(buf561, (128, ), (1, ), 0), reinterpret_tensor(buf552, (128, ), (1, ), 0), reinterpret_tensor(buf551, (128, ), (1, ), 0), reinterpret_tensor(buf547, (512, ), (1, ), 0), reinterpret_tensor(buf545, (512, ), (1, ), 0), reinterpret_tensor(buf540, (128, ), (1, ), 0), reinterpret_tensor(buf539, (128, ), (1, ), 0), reinterpret_tensor(buf534, (128, ), (1, ), 0), reinterpret_tensor(buf533, (128, ), (1, ), 0), reinterpret_tensor(buf515, (128, ), (1, ), 0), reinterpret_tensor(buf514, (128, ), (1, ), 0), reinterpret_tensor(buf505, (128, ), (1, ), 0), reinterpret_tensor(buf504, (128, ), (1, ), 0), reinterpret_tensor(buf495, (128, ), (1, ), 0), reinterpret_tensor(buf494, (128, ), (1, ), 0), reinterpret_tensor(buf485, (128, ), (1, ), 0), reinterpret_tensor(buf484, (128, ), (1, ), 0), reinterpret_tensor(buf475, (128, ), (1, ), 0), reinterpret_tensor(buf474, (128, ), (1, ), 0), reinterpret_tensor(buf470, (512, ), (1, ), 0), reinterpret_tensor(buf468, (512, ), (1, ), 0), reinterpret_tensor(buf463, (128, ), (1, ), 0), reinterpret_tensor(buf462, (128, ), (1, ), 0), reinterpret_tensor(buf457, (128, ), (1, ), 0), reinterpret_tensor(buf456, (128, ), (1, ), 0), reinterpret_tensor(buf438, (128, ), (1, ), 0), reinterpret_tensor(buf437, (128, ), (1, ), 0), reinterpret_tensor(buf428, (128, ), (1, ), 0), reinterpret_tensor(buf427, (128, ), (1, ), 0), reinterpret_tensor(buf418, (128, ), (1, ), 0), reinterpret_tensor(buf417, (128, ), (1, ), 0), reinterpret_tensor(buf408, (128, ), (1, ), 0), reinterpret_tensor(buf407, (128, ), (1, ), 0), reinterpret_tensor(buf398, (128, ), (1, ), 0), reinterpret_tensor(buf397, (128, ), (1, ), 0), reinterpret_tensor(buf393, (512, ), (1, ), 0), reinterpret_tensor(buf391, (512, ), (1, ), 0), reinterpret_tensor(buf386, (128, ), (1, ), 0), reinterpret_tensor(buf385, (128, ), (1, ), 0), reinterpret_tensor(buf380, (128, ), (1, ), 0), reinterpret_tensor(buf379, (128, ), (1, ), 0), reinterpret_tensor(buf361, (128, ), (1, ), 0), reinterpret_tensor(buf360, (128, ), (1, ), 0), reinterpret_tensor(buf351, (128, ), (1, ), 0), reinterpret_tensor(buf350, (128, ), (1, ), 0), reinterpret_tensor(buf341, (128, ), (1, ), 0), reinterpret_tensor(buf340, (128, ), (1, ), 0), reinterpret_tensor(buf331, (128, ), (1, ), 0), reinterpret_tensor(buf330, (128, ), (1, ), 0), reinterpret_tensor(buf321, (128, ), (1, ), 0), reinterpret_tensor(buf320, (128, ), (1, ), 0), reinterpret_tensor(buf316, (512, ), (1, ), 0), reinterpret_tensor(buf314, (512, ), (1, ), 0), reinterpret_tensor(buf309, (128, ), (1, ), 0), reinterpret_tensor(buf308, (128, ), (1, ), 0), reinterpret_tensor(buf303, (128, ), (1, ), 0), reinterpret_tensor(buf302, (128, ), (1, ), 0), reinterpret_tensor(buf284, (128, ), (1, ), 0), reinterpret_tensor(buf283, (128, ), (1, ), 0), reinterpret_tensor(buf274, (128, ), (1, ), 0), reinterpret_tensor(buf273, (128, ), (1, ), 0), reinterpret_tensor(buf264, (128, ), (1, ), 0), reinterpret_tensor(buf263, (128, ), (1, ), 0), reinterpret_tensor(buf254, (128, ), (1, ), 0), reinterpret_tensor(buf253, (128, ), (1, ), 0), reinterpret_tensor(buf244, (128, ), (1, ), 0), reinterpret_tensor(buf243, (128, ), (1, ), 0), reinterpret_tensor(buf239, (512, ), (1, ), 0), reinterpret_tensor(buf237, (512, ), (1, ), 0), reinterpret_tensor(buf232, (128, ), (1, ), 0), reinterpret_tensor(buf231, (128, ), (1, ), 0), reinterpret_tensor(buf226, (128, ), (1, ), 0), reinterpret_tensor(buf225, (128, ), (1, ), 0), reinterpret_tensor(buf207, (128, ), (1, ), 0), reinterpret_tensor(buf206, (128, ), (1, ), 0), reinterpret_tensor(buf197, (128, ), (1, ), 0), reinterpret_tensor(buf196, (128, ), (1, ), 0), reinterpret_tensor(buf187, (128, ), (1, ), 0), reinterpret_tensor(buf186, (128, ), (1, ), 0), reinterpret_tensor(buf177, (128, ), (1, ), 0), reinterpret_tensor(buf176, (128, ), (1, ), 0), reinterpret_tensor(buf167, (128, ), (1, ), 0), reinterpret_tensor(buf166, (128, ), (1, ), 0), reinterpret_tensor(buf162, (512, ), (1, ), 0), reinterpret_tensor(buf160, (512, ), (1, ), 0), reinterpret_tensor(buf155, (128, ), (1, ), 0), reinterpret_tensor(buf154, (128, ), (1, ), 0), reinterpret_tensor(buf149, (128, ), (1, ), 0), reinterpret_tensor(buf148, (128, ), (1, ), 0), reinterpret_tensor(buf130, (128, ), (1, ), 0), reinterpret_tensor(buf129, (128, ), (1, ), 0), reinterpret_tensor(buf120, (128, ), (1, ), 0), reinterpret_tensor(buf119, (128, ), (1, ), 0), reinterpret_tensor(buf110, (128, ), (1, ), 0), reinterpret_tensor(buf109, (128, ), (1, ), 0), reinterpret_tensor(buf100, (128, ), (1, ), 0), reinterpret_tensor(buf99, (128, ), (1, ), 0), reinterpret_tensor(buf90, (128, ), (1, ), 0), reinterpret_tensor(buf89, (128, ), (1, ), 0), reinterpret_tensor(buf84, (512, ), (1, ), 0), reinterpret_tensor(buf83, (512, ), (1, ), 0), reinterpret_tensor(buf73, (30522, 128), (1, 30522), 0), reinterpret_tensor(buf73, (384, 30522), (30522, 1), 3906816), reinterpret_tensor(buf72, (30522, ), (1, ), 0), buf1945, reinterpret_tensor(buf1943, (512, 384), (384, 1), 0), reinterpret_tensor(buf1944, (512, ), (1, ), 0), buf1938, buf1934, reinterpret_tensor(buf1929, (128, 512), (512, 1), 0), reinterpret_tensor(buf1930, (128, ), (1, ), 0), reinterpret_tensor(buf1923, (128, 512), (512, 1), 0), reinterpret_tensor(buf1924, (128, ), (1, ), 0), reinterpret_tensor(buf1917, (128, 128), (128, 1), 0), reinterpret_tensor(buf1918, (128, ), (1, ), 0), reinterpret_tensor(buf1914, (128, 128), (128, 1), 0), reinterpret_tensor(buf1915, (128, ), (1, ), 0), reinterpret_tensor(buf1911, (128, 512), (512, 1), 0), reinterpret_tensor(buf1912, (128, ), (1, ), 0), reinterpret_tensor(buf1904, (128, 128), (128, 1), 0), reinterpret_tensor(buf1905, (128, ), (1, ), 0), reinterpret_tensor(buf1898, (512, 128), (128, 1), 0), reinterpret_tensor(buf1899, (512, ), (1, ), 0), reinterpret_tensor(buf1894, (128, 512), (512, 1), 0), reinterpret_tensor(buf1895, (128, ), (1, ), 0), reinterpret_tensor(buf1888, (512, 128), (128, 1), 0), reinterpret_tensor(buf1889, (512, ), (1, ), 0), reinterpret_tensor(buf1884, (128, 512), (512, 1), 0), reinterpret_tensor(buf1885, (128, ), (1, ), 0), reinterpret_tensor(buf1878, (512, 128), (128, 1), 0), reinterpret_tensor(buf1879, (512, ), (1, ), 0), reinterpret_tensor(buf1874, (128, 512), (512, 1), 0), reinterpret_tensor(buf1875, (128, ), (1, ), 0), reinterpret_tensor(buf1868, (512, 128), (128, 1), 0), reinterpret_tensor(buf1869, (512, ), (1, ), 0), reinterpret_tensor(buf1864, (128, 512), (512, 1), 0), reinterpret_tensor(buf1865, (128, ), (1, ), 0), reinterpret_tensor(buf1858, (512, 128), (128, 1), 0), reinterpret_tensor(buf1859, (512, ), (1, ), 0), reinterpret_tensor(buf1852, (128, 512), (512, 1), 0), reinterpret_tensor(buf1853, (128, ), (1, ), 0), reinterpret_tensor(buf1846, (128, 512), (512, 1), 0), reinterpret_tensor(buf1847, (128, ), (1, ), 0), reinterpret_tensor(buf1840, (128, 128), (128, 1), 0), reinterpret_tensor(buf1841, (128, ), (1, ), 0), reinterpret_tensor(buf1837, (128, 128), (128, 1), 0), reinterpret_tensor(buf1838, (128, ), (1, ), 0), reinterpret_tensor(buf1834, (128, 512), (512, 1), 0), reinterpret_tensor(buf1835, (128, ), (1, ), 0), reinterpret_tensor(buf1827, (128, 128), (128, 1), 0), reinterpret_tensor(buf1828, (128, ), (1, ), 0), reinterpret_tensor(buf1821, (512, 128), (128, 1), 0), reinterpret_tensor(buf1822, (512, ), (1, ), 0), reinterpret_tensor(buf1817, (128, 512), (512, 1), 0), reinterpret_tensor(buf1818, (128, ), (1, ), 0), reinterpret_tensor(buf1811, (512, 128), (128, 1), 0), reinterpret_tensor(buf1812, (512, ), (1, ), 0), reinterpret_tensor(buf1807, (128, 512), (512, 1), 0), reinterpret_tensor(buf1808, (128, ), (1, ), 0), reinterpret_tensor(buf1801, (512, 128), (128, 1), 0), reinterpret_tensor(buf1802, (512, ), (1, ), 0), reinterpret_tensor(buf1797, (128, 512), (512, 1), 0), reinterpret_tensor(buf1798, (128, ), (1, ), 0), reinterpret_tensor(buf1791, (512, 128), (128, 1), 0), reinterpret_tensor(buf1792, (512, ), (1, ), 0), reinterpret_tensor(buf1787, (128, 512), (512, 1), 0), reinterpret_tensor(buf1788, (128, ), (1, ), 0), reinterpret_tensor(buf1781, (512, 128), (128, 1), 0), reinterpret_tensor(buf1782, (512, ), (1, ), 0), reinterpret_tensor(buf1775, (128, 512), (512, 1), 0), reinterpret_tensor(buf1776, (128, ), (1, ), 0), reinterpret_tensor(buf1769, (128, 512), (512, 1), 0), reinterpret_tensor(buf1770, (128, ), (1, ), 0), reinterpret_tensor(buf1763, (128, 128), (128, 1), 0), reinterpret_tensor(buf1764, (128, ), (1, ), 0), reinterpret_tensor(buf1760, (128, 128), (128, 1), 0), reinterpret_tensor(buf1761, (128, ), (1, ), 0), reinterpret_tensor(buf1757, (128, 512), (512, 1), 0), reinterpret_tensor(buf1758, (128, ), (1, ), 0), reinterpret_tensor(buf1750, (128, 128), (128, 1), 0), reinterpret_tensor(buf1751, (128, ), (1, ), 0), reinterpret_tensor(buf1744, (512, 128), (128, 1), 0), reinterpret_tensor(buf1745, (512, ), (1, ), 0), reinterpret_tensor(buf1740, (128, 512), (512, 1), 0), reinterpret_tensor(buf1741, (128, ), (1, ), 0), reinterpret_tensor(buf1734, (512, 128), (128, 1), 0), reinterpret_tensor(buf1735, (512, ), (1, ), 0), reinterpret_tensor(buf1730, (128, 512), (512, 1), 0), reinterpret_tensor(buf1731, (128, ), (1, ), 0), reinterpret_tensor(buf1724, (512, 128), (128, 1), 0), reinterpret_tensor(buf1725, (512, ), (1, ), 0), reinterpret_tensor(buf1720, (128, 512), (512, 1), 0), reinterpret_tensor(buf1721, (128, ), (1, ), 0), reinterpret_tensor(buf1714, (512, 128), (128, 1), 0), reinterpret_tensor(buf1715, (512, ), (1, ), 0), reinterpret_tensor(buf1710, (128, 512), (512, 1), 0), reinterpret_tensor(buf1711, (128, ), (1, ), 0), reinterpret_tensor(buf1704, (512, 128), (128, 1), 0), reinterpret_tensor(buf1705, (512, ), (1, ), 0), reinterpret_tensor(buf1698, (128, 512), (512, 1), 0), reinterpret_tensor(buf1699, (128, ), (1, ), 0), reinterpret_tensor(buf1692, (128, 512), (512, 1), 0), reinterpret_tensor(buf1693, (128, ), (1, ), 0), reinterpret_tensor(buf1686, (128, 128), (128, 1), 0), reinterpret_tensor(buf1687, (128, ), (1, ), 0), reinterpret_tensor(buf1683, (128, 128), (128, 1), 0), reinterpret_tensor(buf1684, (128, ), (1, ), 0), reinterpret_tensor(buf1680, (128, 512), (512, 1), 0), reinterpret_tensor(buf1681, (128, ), (1, ), 0), reinterpret_tensor(buf1673, (128, 128), (128, 1), 0), reinterpret_tensor(buf1674, (128, ), (1, ), 0), reinterpret_tensor(buf1667, (512, 128), (128, 1), 0), reinterpret_tensor(buf1668, (512, ), (1, ), 0), reinterpret_tensor(buf1663, (128, 512), (512, 1), 0), reinterpret_tensor(buf1664, (128, ), (1, ), 0), reinterpret_tensor(buf1657, (512, 128), (128, 1), 0), reinterpret_tensor(buf1658, (512, ), (1, ), 0), reinterpret_tensor(buf1653, (128, 512), (512, 1), 0), reinterpret_tensor(buf1654, (128, ), (1, ), 0), reinterpret_tensor(buf1647, (512, 128), (128, 1), 0), reinterpret_tensor(buf1648, (512, ), (1, ), 0), reinterpret_tensor(buf1643, (128, 512), (512, 1), 0), reinterpret_tensor(buf1644, (128, ), (1, ), 0), reinterpret_tensor(buf1637, (512, 128), (128, 1), 0), reinterpret_tensor(buf1638, (512, ), (1, ), 0), reinterpret_tensor(buf1633, (128, 512), (512, 1), 0), reinterpret_tensor(buf1634, (128, ), (1, ), 0), reinterpret_tensor(buf1627, (512, 128), (128, 1), 0), reinterpret_tensor(buf1628, (512, ), (1, ), 0), reinterpret_tensor(buf1621, (128, 512), (512, 1), 0), reinterpret_tensor(buf1622, (128, ), (1, ), 0), reinterpret_tensor(buf1615, (128, 512), (512, 1), 0), reinterpret_tensor(buf1616, (128, ), (1, ), 0), reinterpret_tensor(buf1609, (128, 128), (128, 1), 0), reinterpret_tensor(buf1610, (128, ), (1, ), 0), reinterpret_tensor(buf1606, (128, 128), (128, 1), 0), reinterpret_tensor(buf1607, (128, ), (1, ), 0), reinterpret_tensor(buf1603, (128, 512), (512, 1), 0), reinterpret_tensor(buf1604, (128, ), (1, ), 0), reinterpret_tensor(buf1596, (128, 128), (128, 1), 0), reinterpret_tensor(buf1597, (128, ), (1, ), 0), reinterpret_tensor(buf1590, (512, 128), (128, 1), 0), reinterpret_tensor(buf1591, (512, ), (1, ), 0), reinterpret_tensor(buf1586, (128, 512), (512, 1), 0), reinterpret_tensor(buf1587, (128, ), (1, ), 0), reinterpret_tensor(buf1580, (512, 128), (128, 1), 0), reinterpret_tensor(buf1581, (512, ), (1, ), 0), reinterpret_tensor(buf1576, (128, 512), (512, 1), 0), reinterpret_tensor(buf1577, (128, ), (1, ), 0), reinterpret_tensor(buf1570, (512, 128), (128, 1), 0), reinterpret_tensor(buf1571, (512, ), (1, ), 0), reinterpret_tensor(buf1566, (128, 512), (512, 1), 0), reinterpret_tensor(buf1567, (128, ), (1, ), 0), reinterpret_tensor(buf1560, (512, 128), (128, 1), 0), reinterpret_tensor(buf1561, (512, ), (1, ), 0), reinterpret_tensor(buf1556, (128, 512), (512, 1), 0), reinterpret_tensor(buf1557, (128, ), (1, ), 0), reinterpret_tensor(buf1550, (512, 128), (128, 1), 0), reinterpret_tensor(buf1551, (512, ), (1, ), 0), reinterpret_tensor(buf1544, (128, 512), (512, 1), 0), reinterpret_tensor(buf1545, (128, ), (1, ), 0), reinterpret_tensor(buf1538, (128, 512), (512, 1), 0), reinterpret_tensor(buf1539, (128, ), (1, ), 0), reinterpret_tensor(buf1532, (128, 128), (128, 1), 0), reinterpret_tensor(buf1533, (128, ), (1, ), 0), reinterpret_tensor(buf1529, (128, 128), (128, 1), 0), reinterpret_tensor(buf1530, (128, ), (1, ), 0), reinterpret_tensor(buf1526, (128, 512), (512, 1), 0), reinterpret_tensor(buf1527, (128, ), (1, ), 0), reinterpret_tensor(buf1519, (128, 128), (128, 1), 0), reinterpret_tensor(buf1520, (128, ), (1, ), 0), reinterpret_tensor(buf1513, (512, 128), (128, 1), 0), reinterpret_tensor(buf1514, (512, ), (1, ), 0), reinterpret_tensor(buf1509, (128, 512), (512, 1), 0), reinterpret_tensor(buf1510, (128, ), (1, ), 0), reinterpret_tensor(buf1503, (512, 128), (128, 1), 0), reinterpret_tensor(buf1504, (512, ), (1, ), 0), reinterpret_tensor(buf1499, (128, 512), (512, 1), 0), reinterpret_tensor(buf1500, (128, ), (1, ), 0), reinterpret_tensor(buf1493, (512, 128), (128, 1), 0), reinterpret_tensor(buf1494, (512, ), (1, ), 0), reinterpret_tensor(buf1489, (128, 512), (512, 1), 0), reinterpret_tensor(buf1490, (128, ), (1, ), 0), reinterpret_tensor(buf1483, (512, 128), (128, 1), 0), reinterpret_tensor(buf1484, (512, ), (1, ), 0), reinterpret_tensor(buf1479, (128, 512), (512, 1), 0), reinterpret_tensor(buf1480, (128, ), (1, ), 0), reinterpret_tensor(buf1473, (512, 128), (128, 1), 0), reinterpret_tensor(buf1474, (512, ), (1, ), 0), reinterpret_tensor(buf1467, (128, 512), (512, 1), 0), reinterpret_tensor(buf1468, (128, ), (1, ), 0), reinterpret_tensor(buf1461, (128, 512), (512, 1), 0), reinterpret_tensor(buf1462, (128, ), (1, ), 0), reinterpret_tensor(buf1455, (128, 128), (128, 1), 0), reinterpret_tensor(buf1456, (128, ), (1, ), 0), reinterpret_tensor(buf1452, (128, 128), (128, 1), 0), reinterpret_tensor(buf1453, (128, ), (1, ), 0), reinterpret_tensor(buf1449, (128, 512), (512, 1), 0), reinterpret_tensor(buf1450, (128, ), (1, ), 0), reinterpret_tensor(buf1442, (128, 128), (128, 1), 0), reinterpret_tensor(buf1443, (128, ), (1, ), 0), reinterpret_tensor(buf1436, (512, 128), (128, 1), 0), reinterpret_tensor(buf1437, (512, ), (1, ), 0), reinterpret_tensor(buf1432, (128, 512), (512, 1), 0), reinterpret_tensor(buf1433, (128, ), (1, ), 0), reinterpret_tensor(buf1426, (512, 128), (128, 1), 0), reinterpret_tensor(buf1427, (512, ), (1, ), 0), reinterpret_tensor(buf1422, (128, 512), (512, 1), 0), reinterpret_tensor(buf1423, (128, ), (1, ), 0), reinterpret_tensor(buf1416, (512, 128), (128, 1), 0), reinterpret_tensor(buf1417, (512, ), (1, ), 0), reinterpret_tensor(buf1412, (128, 512), (512, 1), 0), reinterpret_tensor(buf1413, (128, ), (1, ), 0), reinterpret_tensor(buf1406, (512, 128), (128, 1), 0), reinterpret_tensor(buf1407, (512, ), (1, ), 0), reinterpret_tensor(buf1402, (128, 512), (512, 1), 0), reinterpret_tensor(buf1403, (128, ), (1, ), 0), reinterpret_tensor(buf1396, (512, 128), (128, 1), 0), reinterpret_tensor(buf1397, (512, ), (1, ), 0), reinterpret_tensor(buf1390, (128, 512), (512, 1), 0), reinterpret_tensor(buf1391, (128, ), (1, ), 0), reinterpret_tensor(buf1384, (128, 512), (512, 1), 0), reinterpret_tensor(buf1385, (128, ), (1, ), 0), reinterpret_tensor(buf1378, (128, 128), (128, 1), 0), reinterpret_tensor(buf1379, (128, ), (1, ), 0), reinterpret_tensor(buf1375, (128, 128), (128, 1), 0), reinterpret_tensor(buf1376, (128, ), (1, ), 0), reinterpret_tensor(buf1372, (128, 512), (512, 1), 0), reinterpret_tensor(buf1373, (128, ), (1, ), 0), reinterpret_tensor(buf1365, (128, 128), (128, 1), 0), reinterpret_tensor(buf1366, (128, ), (1, ), 0), reinterpret_tensor(buf1359, (512, 128), (128, 1), 0), reinterpret_tensor(buf1360, (512, ), (1, ), 0), reinterpret_tensor(buf1355, (128, 512), (512, 1), 0), reinterpret_tensor(buf1356, (128, ), (1, ), 0), reinterpret_tensor(buf1349, (512, 128), (128, 1), 0), reinterpret_tensor(buf1350, (512, ), (1, ), 0), reinterpret_tensor(buf1345, (128, 512), (512, 1), 0), reinterpret_tensor(buf1346, (128, ), (1, ), 0), reinterpret_tensor(buf1339, (512, 128), (128, 1), 0), reinterpret_tensor(buf1340, (512, ), (1, ), 0), reinterpret_tensor(buf1335, (128, 512), (512, 1), 0), reinterpret_tensor(buf1336, (128, ), (1, ), 0), reinterpret_tensor(buf1329, (512, 128), (128, 1), 0), reinterpret_tensor(buf1330, (512, ), (1, ), 0), reinterpret_tensor(buf1325, (128, 512), (512, 1), 0), reinterpret_tensor(buf1326, (128, ), (1, ), 0), reinterpret_tensor(buf1319, (512, 128), (128, 1), 0), reinterpret_tensor(buf1320, (512, ), (1, ), 0), reinterpret_tensor(buf1313, (128, 512), (512, 1), 0), reinterpret_tensor(buf1314, (128, ), (1, ), 0), reinterpret_tensor(buf1307, (128, 512), (512, 1), 0), reinterpret_tensor(buf1308, (128, ), (1, ), 0), reinterpret_tensor(buf1301, (128, 128), (128, 1), 0), reinterpret_tensor(buf1302, (128, ), (1, ), 0), reinterpret_tensor(buf1298, (128, 128), (128, 1), 0), reinterpret_tensor(buf1299, (128, ), (1, ), 0), reinterpret_tensor(buf1295, (128, 512), (512, 1), 0), reinterpret_tensor(buf1296, (128, ), (1, ), 0), reinterpret_tensor(buf1288, (128, 128), (128, 1), 0), reinterpret_tensor(buf1289, (128, ), (1, ), 0), reinterpret_tensor(buf1282, (512, 128), (128, 1), 0), reinterpret_tensor(buf1283, (512, ), (1, ), 0), reinterpret_tensor(buf1278, (128, 512), (512, 1), 0), reinterpret_tensor(buf1279, (128, ), (1, ), 0), reinterpret_tensor(buf1272, (512, 128), (128, 1), 0), reinterpret_tensor(buf1273, (512, ), (1, ), 0), reinterpret_tensor(buf1268, (128, 512), (512, 1), 0), reinterpret_tensor(buf1269, (128, ), (1, ), 0), reinterpret_tensor(buf1262, (512, 128), (128, 1), 0), reinterpret_tensor(buf1263, (512, ), (1, ), 0), reinterpret_tensor(buf1258, (128, 512), (512, 1), 0), reinterpret_tensor(buf1259, (128, ), (1, ), 0), reinterpret_tensor(buf1252, (512, 128), (128, 1), 0), reinterpret_tensor(buf1253, (512, ), (1, ), 0), reinterpret_tensor(buf1248, (128, 512), (512, 1), 0), reinterpret_tensor(buf1249, (128, ), (1, ), 0), reinterpret_tensor(buf1242, (512, 128), (128, 1), 0), reinterpret_tensor(buf1243, (512, ), (1, ), 0), reinterpret_tensor(buf1236, (128, 512), (512, 1), 0), reinterpret_tensor(buf1237, (128, ), (1, ), 0), reinterpret_tensor(buf1230, (128, 512), (512, 1), 0), reinterpret_tensor(buf1231, (128, ), (1, ), 0), reinterpret_tensor(buf1224, (128, 128), (128, 1), 0), reinterpret_tensor(buf1225, (128, ), (1, ), 0), reinterpret_tensor(buf1221, (128, 128), (128, 1), 0), reinterpret_tensor(buf1222, (128, ), (1, ), 0), reinterpret_tensor(buf1218, (128, 512), (512, 1), 0), reinterpret_tensor(buf1219, (128, ), (1, ), 0), reinterpret_tensor(buf1211, (128, 128), (128, 1), 0), reinterpret_tensor(buf1212, (128, ), (1, ), 0), reinterpret_tensor(buf1205, (512, 128), (128, 1), 0), reinterpret_tensor(buf1206, (512, ), (1, ), 0), reinterpret_tensor(buf1201, (128, 512), (512, 1), 0), reinterpret_tensor(buf1202, (128, ), (1, ), 0), reinterpret_tensor(buf1195, (512, 128), (128, 1), 0), reinterpret_tensor(buf1196, (512, ), (1, ), 0), reinterpret_tensor(buf1191, (128, 512), (512, 1), 0), reinterpret_tensor(buf1192, (128, ), (1, ), 0), reinterpret_tensor(buf1185, (512, 128), (128, 1), 0), reinterpret_tensor(buf1186, (512, ), (1, ), 0), reinterpret_tensor(buf1181, (128, 512), (512, 1), 0), reinterpret_tensor(buf1182, (128, ), (1, ), 0), reinterpret_tensor(buf1175, (512, 128), (128, 1), 0), reinterpret_tensor(buf1176, (512, ), (1, ), 0), reinterpret_tensor(buf1171, (128, 512), (512, 1), 0), reinterpret_tensor(buf1172, (128, ), (1, ), 0), reinterpret_tensor(buf1165, (512, 128), (128, 1), 0), reinterpret_tensor(buf1166, (512, ), (1, ), 0), reinterpret_tensor(buf1159, (128, 512), (512, 1), 0), reinterpret_tensor(buf1160, (128, ), (1, ), 0), reinterpret_tensor(buf1153, (128, 512), (512, 1), 0), reinterpret_tensor(buf1154, (128, ), (1, ), 0), reinterpret_tensor(buf1147, (128, 128), (128, 1), 0), reinterpret_tensor(buf1148, (128, ), (1, ), 0), reinterpret_tensor(buf1144, (128, 128), (128, 1), 0), reinterpret_tensor(buf1145, (128, ), (1, ), 0), reinterpret_tensor(buf1141, (128, 512), (512, 1), 0), reinterpret_tensor(buf1142, (128, ), (1, ), 0), reinterpret_tensor(buf1134, (128, 128), (128, 1), 0), reinterpret_tensor(buf1135, (128, ), (1, ), 0), reinterpret_tensor(buf1128, (512, 128), (128, 1), 0), reinterpret_tensor(buf1129, (512, ), (1, ), 0), reinterpret_tensor(buf1124, (128, 512), (512, 1), 0), reinterpret_tensor(buf1125, (128, ), (1, ), 0), reinterpret_tensor(buf1118, (512, 128), (128, 1), 0), reinterpret_tensor(buf1119, (512, ), (1, ), 0), reinterpret_tensor(buf1114, (128, 512), (512, 1), 0), reinterpret_tensor(buf1115, (128, ), (1, ), 0), reinterpret_tensor(buf1108, (512, 128), (128, 1), 0), reinterpret_tensor(buf1109, (512, ), (1, ), 0), reinterpret_tensor(buf1104, (128, 512), (512, 1), 0), reinterpret_tensor(buf1105, (128, ), (1, ), 0), reinterpret_tensor(buf1098, (512, 128), (128, 1), 0), reinterpret_tensor(buf1099, (512, ), (1, ), 0), reinterpret_tensor(buf1094, (128, 512), (512, 1), 0), reinterpret_tensor(buf1095, (128, ), (1, ), 0), reinterpret_tensor(buf1088, (512, 128), (128, 1), 0), reinterpret_tensor(buf1089, (512, ), (1, ), 0), reinterpret_tensor(buf1082, (128, 512), (512, 1), 0), reinterpret_tensor(buf1083, (128, ), (1, ), 0), reinterpret_tensor(buf1076, (128, 512), (512, 1), 0), reinterpret_tensor(buf1077, (128, ), (1, ), 0), reinterpret_tensor(buf1070, (128, 128), (128, 1), 0), reinterpret_tensor(buf1071, (128, ), (1, ), 0), reinterpret_tensor(buf1067, (128, 128), (128, 1), 0), reinterpret_tensor(buf1068, (128, ), (1, ), 0), reinterpret_tensor(buf1064, (128, 512), (512, 1), 0), reinterpret_tensor(buf1065, (128, ), (1, ), 0), reinterpret_tensor(buf1057, (128, 128), (128, 1), 0), reinterpret_tensor(buf1058, (128, ), (1, ), 0), reinterpret_tensor(buf1051, (512, 128), (128, 1), 0), reinterpret_tensor(buf1052, (512, ), (1, ), 0), reinterpret_tensor(buf1047, (128, 512), (512, 1), 0), reinterpret_tensor(buf1048, (128, ), (1, ), 0), reinterpret_tensor(buf1041, (512, 128), (128, 1), 0), reinterpret_tensor(buf1042, (512, ), (1, ), 0), reinterpret_tensor(buf1037, (128, 512), (512, 1), 0), reinterpret_tensor(buf1038, (128, ), (1, ), 0), reinterpret_tensor(buf1031, (512, 128), (128, 1), 0), reinterpret_tensor(buf1032, (512, ), (1, ), 0), reinterpret_tensor(buf1027, (128, 512), (512, 1), 0), reinterpret_tensor(buf1028, (128, ), (1, ), 0), reinterpret_tensor(buf1021, (512, 128), (128, 1), 0), reinterpret_tensor(buf1022, (512, ), (1, ), 0), reinterpret_tensor(buf1017, (128, 512), (512, 1), 0), reinterpret_tensor(buf1018, (128, ), (1, ), 0), reinterpret_tensor(buf1011, (512, 128), (128, 1), 0), reinterpret_tensor(buf1012, (512, ), (1, ), 0), reinterpret_tensor(buf1005, (128, 512), (512, 1), 0), reinterpret_tensor(buf1006, (128, ), (1, ), 0), reinterpret_tensor(buf999, (128, 512), (512, 1), 0), reinterpret_tensor(buf1000, (128, ), (1, ), 0), reinterpret_tensor(buf993, (128, 128), (128, 1), 0), reinterpret_tensor(buf994, (128, ), (1, ), 0), reinterpret_tensor(buf990, (128, 128), (128, 1), 0), reinterpret_tensor(buf991, (128, ), (1, ), 0), reinterpret_tensor(buf987, (128, 512), (512, 1), 0), reinterpret_tensor(buf988, (128, ), (1, ), 0), reinterpret_tensor(buf980, (128, 128), (128, 1), 0), reinterpret_tensor(buf981, (128, ), (1, ), 0), reinterpret_tensor(buf974, (512, 128), (128, 1), 0), reinterpret_tensor(buf975, (512, ), (1, ), 0), reinterpret_tensor(buf970, (128, 512), (512, 1), 0), reinterpret_tensor(buf971, (128, ), (1, ), 0), reinterpret_tensor(buf964, (512, 128), (128, 1), 0), reinterpret_tensor(buf965, (512, ), (1, ), 0), reinterpret_tensor(buf960, (128, 512), (512, 1), 0), reinterpret_tensor(buf961, (128, ), (1, ), 0), reinterpret_tensor(buf954, (512, 128), (128, 1), 0), reinterpret_tensor(buf955, (512, ), (1, ), 0), reinterpret_tensor(buf950, (128, 512), (512, 1), 0), reinterpret_tensor(buf951, (128, ), (1, ), 0), reinterpret_tensor(buf944, (512, 128), (128, 1), 0), reinterpret_tensor(buf945, (512, ), (1, ), 0), reinterpret_tensor(buf940, (128, 512), (512, 1), 0), reinterpret_tensor(buf941, (128, ), (1, ), 0), reinterpret_tensor(buf934, (512, 128), (128, 1), 0), reinterpret_tensor(buf935, (512, ), (1, ), 0), reinterpret_tensor(buf928, (128, 512), (512, 1), 0), reinterpret_tensor(buf929, (128, ), (1, ), 0), reinterpret_tensor(buf922, (128, 512), (512, 1), 0), reinterpret_tensor(buf923, (128, ), (1, ), 0), reinterpret_tensor(buf916, (128, 128), (128, 1), 0), reinterpret_tensor(buf917, (128, ), (1, ), 0), reinterpret_tensor(buf913, (128, 128), (128, 1), 0), reinterpret_tensor(buf914, (128, ), (1, ), 0), reinterpret_tensor(buf910, (128, 512), (512, 1), 0), reinterpret_tensor(buf911, (128, ), (1, ), 0), reinterpret_tensor(buf903, (128, 128), (128, 1), 0), reinterpret_tensor(buf904, (128, ), (1, ), 0), reinterpret_tensor(buf897, (512, 128), (128, 1), 0), reinterpret_tensor(buf898, (512, ), (1, ), 0), reinterpret_tensor(buf893, (128, 512), (512, 1), 0), reinterpret_tensor(buf894, (128, ), (1, ), 0), reinterpret_tensor(buf887, (512, 128), (128, 1), 0), reinterpret_tensor(buf888, (512, ), (1, ), 0), reinterpret_tensor(buf883, (128, 512), (512, 1), 0), reinterpret_tensor(buf884, (128, ), (1, ), 0), reinterpret_tensor(buf877, (512, 128), (128, 1), 0), reinterpret_tensor(buf878, (512, ), (1, ), 0), reinterpret_tensor(buf873, (128, 512), (512, 1), 0), reinterpret_tensor(buf874, (128, ), (1, ), 0), reinterpret_tensor(buf867, (512, 128), (128, 1), 0), reinterpret_tensor(buf868, (512, ), (1, ), 0), reinterpret_tensor(buf863, (128, 512), (512, 1), 0), reinterpret_tensor(buf864, (128, ), (1, ), 0), reinterpret_tensor(buf857, (512, 128), (128, 1), 0), reinterpret_tensor(buf858, (512, ), (1, ), 0), reinterpret_tensor(buf851, (128, 512), (512, 1), 0), reinterpret_tensor(buf852, (128, ), (1, ), 0), reinterpret_tensor(buf845, (128, 512), (512, 1), 0), reinterpret_tensor(buf846, (128, ), (1, ), 0), reinterpret_tensor(buf839, (128, 128), (128, 1), 0), reinterpret_tensor(buf840, (128, ), (1, ), 0), reinterpret_tensor(buf836, (128, 128), (128, 1), 0), reinterpret_tensor(buf837, (128, ), (1, ), 0), reinterpret_tensor(buf833, (128, 512), (512, 1), 0), reinterpret_tensor(buf834, (128, ), (1, ), 0), reinterpret_tensor(buf826, (128, 128), (128, 1), 0), reinterpret_tensor(buf827, (128, ), (1, ), 0), reinterpret_tensor(buf820, (512, 128), (128, 1), 0), reinterpret_tensor(buf821, (512, ), (1, ), 0), reinterpret_tensor(buf816, (128, 512), (512, 1), 0), reinterpret_tensor(buf817, (128, ), (1, ), 0), reinterpret_tensor(buf810, (512, 128), (128, 1), 0), reinterpret_tensor(buf811, (512, ), (1, ), 0), reinterpret_tensor(buf806, (128, 512), (512, 1), 0), reinterpret_tensor(buf807, (128, ), (1, ), 0), reinterpret_tensor(buf800, (512, 128), (128, 1), 0), reinterpret_tensor(buf801, (512, ), (1, ), 0), reinterpret_tensor(buf796, (128, 512), (512, 1), 0), reinterpret_tensor(buf797, (128, ), (1, ), 0), reinterpret_tensor(buf790, (512, 128), (128, 1), 0), reinterpret_tensor(buf791, (512, ), (1, ), 0), reinterpret_tensor(buf786, (128, 512), (512, 1), 0), reinterpret_tensor(buf787, (128, ), (1, ), 0), reinterpret_tensor(buf780, (512, 128), (128, 1), 0), reinterpret_tensor(buf781, (512, ), (1, ), 0), reinterpret_tensor(buf774, (128, 512), (512, 1), 0), reinterpret_tensor(buf775, (128, ), (1, ), 0), reinterpret_tensor(buf768, (128, 512), (512, 1), 0), reinterpret_tensor(buf769, (128, ), (1, ), 0), reinterpret_tensor(buf762, (128, 128), (128, 1), 0), reinterpret_tensor(buf763, (128, ), (1, ), 0), reinterpret_tensor(buf759, (128, 128), (128, 1), 0), reinterpret_tensor(buf760, (128, ), (1, ), 0), reinterpret_tensor(buf756, (128, 512), (512, 1), 0), reinterpret_tensor(buf757, (128, ), (1, ), 0), reinterpret_tensor(buf749, (128, 128), (128, 1), 0), reinterpret_tensor(buf750, (128, ), (1, ), 0), reinterpret_tensor(buf743, (512, 128), (128, 1), 0), reinterpret_tensor(buf744, (512, ), (1, ), 0), reinterpret_tensor(buf739, (128, 512), (512, 1), 0), reinterpret_tensor(buf740, (128, ), (1, ), 0), reinterpret_tensor(buf733, (512, 128), (128, 1), 0), reinterpret_tensor(buf734, (512, ), (1, ), 0), reinterpret_tensor(buf729, (128, 512), (512, 1), 0), reinterpret_tensor(buf730, (128, ), (1, ), 0), reinterpret_tensor(buf723, (512, 128), (128, 1), 0), reinterpret_tensor(buf724, (512, ), (1, ), 0), reinterpret_tensor(buf719, (128, 512), (512, 1), 0), reinterpret_tensor(buf720, (128, ), (1, ), 0), reinterpret_tensor(buf713, (512, 128), (128, 1), 0), reinterpret_tensor(buf714, (512, ), (1, ), 0), reinterpret_tensor(buf709, (128, 512), (512, 1), 0), reinterpret_tensor(buf710, (128, ), (1, ), 0), reinterpret_tensor(buf703, (512, 128), (128, 1), 0), reinterpret_tensor(buf704, (512, ), (1, ), 0), reinterpret_tensor(buf697, (128, 512), (512, 1), 0), reinterpret_tensor(buf698, (128, ), (1, ), 0), reinterpret_tensor(buf691, (128, 512), (512, 1), 0), reinterpret_tensor(buf692, (128, ), (1, ), 0), reinterpret_tensor(buf685, (128, 128), (128, 1), 0), reinterpret_tensor(buf686, (128, ), (1, ), 0), reinterpret_tensor(buf682, (128, 128), (128, 1), 0), reinterpret_tensor(buf683, (128, ), (1, ), 0), reinterpret_tensor(buf679, (128, 512), (512, 1), 0), reinterpret_tensor(buf680, (128, ), (1, ), 0), reinterpret_tensor(buf672, (128, 128), (128, 1), 0), reinterpret_tensor(buf673, (128, ), (1, ), 0), reinterpret_tensor(buf666, (512, 128), (128, 1), 0), reinterpret_tensor(buf667, (512, ), (1, ), 0), reinterpret_tensor(buf662, (128, 512), (512, 1), 0), reinterpret_tensor(buf663, (128, ), (1, ), 0), reinterpret_tensor(buf656, (512, 128), (128, 1), 0), reinterpret_tensor(buf657, (512, ), (1, ), 0), reinterpret_tensor(buf652, (128, 512), (512, 1), 0), reinterpret_tensor(buf653, (128, ), (1, ), 0), reinterpret_tensor(buf646, (512, 128), (128, 1), 0), reinterpret_tensor(buf647, (512, ), (1, ), 0), reinterpret_tensor(buf642, (128, 512), (512, 1), 0), reinterpret_tensor(buf643, (128, ), (1, ), 0), reinterpret_tensor(buf636, (512, 128), (128, 1), 0), reinterpret_tensor(buf637, (512, ), (1, ), 0), reinterpret_tensor(buf632, (128, 512), (512, 1), 0), reinterpret_tensor(buf633, (128, ), (1, ), 0), reinterpret_tensor(buf626, (512, 128), (128, 1), 0), reinterpret_tensor(buf627, (512, ), (1, ), 0), reinterpret_tensor(buf620, (128, 512), (512, 1), 0), reinterpret_tensor(buf621, (128, ), (1, ), 0), reinterpret_tensor(buf614, (128, 512), (512, 1), 0), reinterpret_tensor(buf615, (128, ), (1, ), 0), reinterpret_tensor(buf608, (128, 128), (128, 1), 0), reinterpret_tensor(buf609, (128, ), (1, ), 0), reinterpret_tensor(buf605, (128, 128), (128, 1), 0), reinterpret_tensor(buf606, (128, ), (1, ), 0), reinterpret_tensor(buf602, (128, 512), (512, 1), 0), reinterpret_tensor(buf603, (128, ), (1, ), 0), reinterpret_tensor(buf595, (128, 128), (128, 1), 0), reinterpret_tensor(buf596, (128, ), (1, ), 0), reinterpret_tensor(buf589, (512, 128), (128, 1), 0), reinterpret_tensor(buf590, (512, ), (1, ), 0), reinterpret_tensor(buf585, (128, 512), (512, 1), 0), reinterpret_tensor(buf586, (128, ), (1, ), 0), reinterpret_tensor(buf579, (512, 128), (128, 1), 0), reinterpret_tensor(buf580, (512, ), (1, ), 0), reinterpret_tensor(buf575, (128, 512), (512, 1), 0), reinterpret_tensor(buf576, (128, ), (1, ), 0), reinterpret_tensor(buf569, (512, 128), (128, 1), 0), reinterpret_tensor(buf570, (512, ), (1, ), 0), reinterpret_tensor(buf565, (128, 512), (512, 1), 0), reinterpret_tensor(buf566, (128, ), (1, ), 0), reinterpret_tensor(buf559, (512, 128), (128, 1), 0), reinterpret_tensor(buf560, (512, ), (1, ), 0), reinterpret_tensor(buf555, (128, 512), (512, 1), 0), reinterpret_tensor(buf556, (128, ), (1, ), 0), reinterpret_tensor(buf549, (512, 128), (128, 1), 0), reinterpret_tensor(buf550, (512, ), (1, ), 0), reinterpret_tensor(buf543, (128, 512), (512, 1), 0), reinterpret_tensor(buf544, (128, ), (1, ), 0), reinterpret_tensor(buf537, (128, 512), (512, 1), 0), reinterpret_tensor(buf538, (128, ), (1, ), 0), reinterpret_tensor(buf531, (128, 128), (128, 1), 0), reinterpret_tensor(buf532, (128, ), (1, ), 0), reinterpret_tensor(buf528, (128, 128), (128, 1), 0), reinterpret_tensor(buf529, (128, ), (1, ), 0), reinterpret_tensor(buf525, (128, 512), (512, 1), 0), reinterpret_tensor(buf526, (128, ), (1, ), 0), reinterpret_tensor(buf518, (128, 128), (128, 1), 0), reinterpret_tensor(buf519, (128, ), (1, ), 0), reinterpret_tensor(buf512, (512, 128), (128, 1), 0), reinterpret_tensor(buf513, (512, ), (1, ), 0), reinterpret_tensor(buf508, (128, 512), (512, 1), 0), reinterpret_tensor(buf509, (128, ), (1, ), 0), reinterpret_tensor(buf502, (512, 128), (128, 1), 0), reinterpret_tensor(buf503, (512, ), (1, ), 0), reinterpret_tensor(buf498, (128, 512), (512, 1), 0), reinterpret_tensor(buf499, (128, ), (1, ), 0), reinterpret_tensor(buf492, (512, 128), (128, 1), 0), reinterpret_tensor(buf493, (512, ), (1, ), 0), reinterpret_tensor(buf488, (128, 512), (512, 1), 0), reinterpret_tensor(buf489, (128, ), (1, ), 0), reinterpret_tensor(buf482, (512, 128), (128, 1), 0), reinterpret_tensor(buf483, (512, ), (1, ), 0), reinterpret_tensor(buf478, (128, 512), (512, 1), 0), reinterpret_tensor(buf479, (128, ), (1, ), 0), reinterpret_tensor(buf472, (512, 128), (128, 1), 0), reinterpret_tensor(buf473, (512, ), (1, ), 0), reinterpret_tensor(buf466, (128, 512), (512, 1), 0), reinterpret_tensor(buf467, (128, ), (1, ), 0), reinterpret_tensor(buf460, (128, 512), (512, 1), 0), reinterpret_tensor(buf461, (128, ), (1, ), 0), reinterpret_tensor(buf454, (128, 128), (128, 1), 0), reinterpret_tensor(buf455, (128, ), (1, ), 0), reinterpret_tensor(buf451, (128, 128), (128, 1), 0), reinterpret_tensor(buf452, (128, ), (1, ), 0), reinterpret_tensor(buf448, (128, 512), (512, 1), 0), reinterpret_tensor(buf449, (128, ), (1, ), 0), reinterpret_tensor(buf441, (128, 128), (128, 1), 0), reinterpret_tensor(buf442, (128, ), (1, ), 0), reinterpret_tensor(buf435, (512, 128), (128, 1), 0), reinterpret_tensor(buf436, (512, ), (1, ), 0), reinterpret_tensor(buf431, (128, 512), (512, 1), 0), reinterpret_tensor(buf432, (128, ), (1, ), 0), reinterpret_tensor(buf425, (512, 128), (128, 1), 0), reinterpret_tensor(buf426, (512, ), (1, ), 0), reinterpret_tensor(buf421, (128, 512), (512, 1), 0), reinterpret_tensor(buf422, (128, ), (1, ), 0), reinterpret_tensor(buf415, (512, 128), (128, 1), 0), reinterpret_tensor(buf416, (512, ), (1, ), 0), reinterpret_tensor(buf411, (128, 512), (512, 1), 0), reinterpret_tensor(buf412, (128, ), (1, ), 0), reinterpret_tensor(buf405, (512, 128), (128, 1), 0), reinterpret_tensor(buf406, (512, ), (1, ), 0), reinterpret_tensor(buf401, (128, 512), (512, 1), 0), reinterpret_tensor(buf402, (128, ), (1, ), 0), reinterpret_tensor(buf395, (512, 128), (128, 1), 0), reinterpret_tensor(buf396, (512, ), (1, ), 0), reinterpret_tensor(buf389, (128, 512), (512, 1), 0), reinterpret_tensor(buf390, (128, ), (1, ), 0), reinterpret_tensor(buf383, (128, 512), (512, 1), 0), reinterpret_tensor(buf384, (128, ), (1, ), 0), reinterpret_tensor(buf377, (128, 128), (128, 1), 0), reinterpret_tensor(buf378, (128, ), (1, ), 0), reinterpret_tensor(buf374, (128, 128), (128, 1), 0), reinterpret_tensor(buf375, (128, ), (1, ), 0), reinterpret_tensor(buf371, (128, 512), (512, 1), 0), reinterpret_tensor(buf372, (128, ), (1, ), 0), reinterpret_tensor(buf364, (128, 128), (128, 1), 0), reinterpret_tensor(buf365, (128, ), (1, ), 0), reinterpret_tensor(buf358, (512, 128), (128, 1), 0), reinterpret_tensor(buf359, (512, ), (1, ), 0), reinterpret_tensor(buf354, (128, 512), (512, 1), 0), reinterpret_tensor(buf355, (128, ), (1, ), 0), reinterpret_tensor(buf348, (512, 128), (128, 1), 0), reinterpret_tensor(buf349, (512, ), (1, ), 0), reinterpret_tensor(buf344, (128, 512), (512, 1), 0), reinterpret_tensor(buf345, (128, ), (1, ), 0), reinterpret_tensor(buf338, (512, 128), (128, 1), 0), reinterpret_tensor(buf339, (512, ), (1, ), 0), reinterpret_tensor(buf334, (128, 512), (512, 1), 0), reinterpret_tensor(buf335, (128, ), (1, ), 0), reinterpret_tensor(buf328, (512, 128), (128, 1), 0), reinterpret_tensor(buf329, (512, ), (1, ), 0), reinterpret_tensor(buf324, (128, 512), (512, 1), 0), reinterpret_tensor(buf325, (128, ), (1, ), 0), reinterpret_tensor(buf318, (512, 128), (128, 1), 0), reinterpret_tensor(buf319, (512, ), (1, ), 0), reinterpret_tensor(buf312, (128, 512), (512, 1), 0), reinterpret_tensor(buf313, (128, ), (1, ), 0), reinterpret_tensor(buf306, (128, 512), (512, 1), 0), reinterpret_tensor(buf307, (128, ), (1, ), 0), reinterpret_tensor(buf300, (128, 128), (128, 1), 0), reinterpret_tensor(buf301, (128, ), (1, ), 0), reinterpret_tensor(buf297, (128, 128), (128, 1), 0), reinterpret_tensor(buf298, (128, ), (1, ), 0), reinterpret_tensor(buf294, (128, 512), (512, 1), 0), reinterpret_tensor(buf295, (128, ), (1, ), 0), reinterpret_tensor(buf287, (128, 128), (128, 1), 0), reinterpret_tensor(buf288, (128, ), (1, ), 0), reinterpret_tensor(buf281, (512, 128), (128, 1), 0), reinterpret_tensor(buf282, (512, ), (1, ), 0), reinterpret_tensor(buf277, (128, 512), (512, 1), 0), reinterpret_tensor(buf278, (128, ), (1, ), 0), reinterpret_tensor(buf271, (512, 128), (128, 1), 0), reinterpret_tensor(buf272, (512, ), (1, ), 0), reinterpret_tensor(buf267, (128, 512), (512, 1), 0), reinterpret_tensor(buf268, (128, ), (1, ), 0), reinterpret_tensor(buf261, (512, 128), (128, 1), 0), reinterpret_tensor(buf262, (512, ), (1, ), 0), reinterpret_tensor(buf257, (128, 512), (512, 1), 0), reinterpret_tensor(buf258, (128, ), (1, ), 0), reinterpret_tensor(buf251, (512, 128), (128, 1), 0), reinterpret_tensor(buf252, (512, ), (1, ), 0), reinterpret_tensor(buf247, (128, 512), (512, 1), 0), reinterpret_tensor(buf248, (128, ), (1, ), 0), reinterpret_tensor(buf241, (512, 128), (128, 1), 0), reinterpret_tensor(buf242, (512, ), (1, ), 0), reinterpret_tensor(buf235, (128, 512), (512, 1), 0), reinterpret_tensor(buf236, (128, ), (1, ), 0), reinterpret_tensor(buf229, (128, 512), (512, 1), 0), reinterpret_tensor(buf230, (128, ), (1, ), 0), reinterpret_tensor(buf223, (128, 128), (128, 1), 0), reinterpret_tensor(buf224, (128, ), (1, ), 0), reinterpret_tensor(buf220, (128, 128), (128, 1), 0), reinterpret_tensor(buf221, (128, ), (1, ), 0), reinterpret_tensor(buf217, (128, 512), (512, 1), 0), reinterpret_tensor(buf218, (128, ), (1, ), 0), reinterpret_tensor(buf210, (128, 128), (128, 1), 0), reinterpret_tensor(buf211, (128, ), (1, ), 0), reinterpret_tensor(buf204, (512, 128), (128, 1), 0), reinterpret_tensor(buf205, (512, ), (1, ), 0), reinterpret_tensor(buf200, (128, 512), (512, 1), 0), reinterpret_tensor(buf201, (128, ), (1, ), 0), reinterpret_tensor(buf194, (512, 128), (128, 1), 0), reinterpret_tensor(buf195, (512, ), (1, ), 0), reinterpret_tensor(buf190, (128, 512), (512, 1), 0), reinterpret_tensor(buf191, (128, ), (1, ), 0), reinterpret_tensor(buf184, (512, 128), (128, 1), 0), reinterpret_tensor(buf185, (512, ), (1, ), 0), reinterpret_tensor(buf180, (128, 512), (512, 1), 0), reinterpret_tensor(buf181, (128, ), (1, ), 0), reinterpret_tensor(buf174, (512, 128), (128, 1), 0), reinterpret_tensor(buf175, (512, ), (1, ), 0), reinterpret_tensor(buf170, (128, 512), (512, 1), 0), reinterpret_tensor(buf171, (128, ), (1, ), 0), reinterpret_tensor(buf164, (512, 128), (128, 1), 0), reinterpret_tensor(buf165, (512, ), (1, ), 0), reinterpret_tensor(buf158, (128, 512), (512, 1), 0), reinterpret_tensor(buf159, (128, ), (1, ), 0), reinterpret_tensor(buf152, (128, 512), (512, 1), 0), reinterpret_tensor(buf153, (128, ), (1, ), 0), reinterpret_tensor(buf146, (128, 128), (128, 1), 0), reinterpret_tensor(buf147, (128, ), (1, ), 0), reinterpret_tensor(buf143, (128, 128), (128, 1), 0), reinterpret_tensor(buf144, (128, ), (1, ), 0), reinterpret_tensor(buf140, (128, 512), (512, 1), 0), reinterpret_tensor(buf141, (128, ), (1, ), 0), reinterpret_tensor(buf133, (128, 128), (128, 1), 0), reinterpret_tensor(buf134, (128, ), (1, ), 0), reinterpret_tensor(buf127, (512, 128), (128, 1), 0), reinterpret_tensor(buf128, (512, ), (1, ), 0), reinterpret_tensor(buf123, (128, 512), (512, 1), 0), reinterpret_tensor(buf124, (128, ), (1, ), 0), reinterpret_tensor(buf117, (512, 128), (128, 1), 0), reinterpret_tensor(buf118, (512, ), (1, ), 0), reinterpret_tensor(buf113, (128, 512), (512, 1), 0), reinterpret_tensor(buf114, (128, ), (1, ), 0), reinterpret_tensor(buf107, (512, 128), (128, 1), 0), reinterpret_tensor(buf108, (512, ), (1, ), 0), reinterpret_tensor(buf103, (128, 512), (512, 1), 0), reinterpret_tensor(buf104, (128, ), (1, ), 0), reinterpret_tensor(buf97, (512, 128), (128, 1), 0), reinterpret_tensor(buf98, (512, ), (1, ), 0), reinterpret_tensor(buf93, (128, 512), (512, 1), 0), reinterpret_tensor(buf94, (128, ), (1, ), 0), reinterpret_tensor(buf87, (512, 128), (128, 1), 0), reinterpret_tensor(buf88, (512, ), (1, ), 0), reinterpret_tensor(buf81, (512, 512), (512, 1), 0), reinterpret_tensor(buf82, (512, ), (1, ), 0), buf77, buf78, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1117 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1120 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    primals_1121 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    full_default = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    slice_4 = rand_strided((1, 128), (512, 1), device='cuda:0', dtype=torch.int64)
    view = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    add_1 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    view_2 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_6 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_69 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_70 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_71 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_212 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_213 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_214 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_47 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_22 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_24 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_26 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_8 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_28 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_30 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_34 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_12 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_36 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_38 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_40 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    add_16 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    view_42 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_17 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_46 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_66 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_67 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_68 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_205 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_206 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_207 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_45 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_62 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_21 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_64 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_66 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_23 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_68 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_70 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_25 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_72 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_74 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_27 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_76 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_78 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_29 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_80 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_30 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_82 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_31 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_32 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_63 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_64 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_65 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_198 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_199 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_200 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_43 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_102 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_36 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_106 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_38 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_40 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_112 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_114 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_42 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_116 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_118 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_44 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_120 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_45 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_122 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_47 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_126 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_60 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_61 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_62 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_191 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_192 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_193 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_41 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_142 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_51 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_144 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_146 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_53 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_148 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_150 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_55 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_154 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_57 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_156 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_158 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_59 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_160 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_60 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_162 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_61 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_62 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_166 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_57 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_58 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_59 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_184 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_185 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_186 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_39 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_182 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_66 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_184 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_186 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_68 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_188 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_190 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_70 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_192 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_194 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_72 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_196 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_198 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_74 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_200 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_75 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_202 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_76 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_77 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_206 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_54 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_55 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_56 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_177 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_178 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_179 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_37 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_222 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_81 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_224 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_226 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_83 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_228 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_230 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_85 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_232 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_234 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_87 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_236 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_238 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_89 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_240 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_90 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_242 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_91 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_92 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_246 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_51 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_52 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_53 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_170 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_171 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_172 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_35 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_262 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_96 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_264 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_266 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_98 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_268 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_270 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_100 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_272 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_274 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_102 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_276 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_278 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_104 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_280 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_105 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_282 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_106 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_107 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_286 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_48 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_49 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_50 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_163 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_164 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_165 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_33 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_302 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_111 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_304 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_306 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_113 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_308 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_310 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_115 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_312 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_314 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_117 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_316 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_318 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_119 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_320 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_120 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_322 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_121 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_122 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_326 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_45 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_46 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_47 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_156 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_157 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_158 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_31 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_342 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_126 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_344 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_346 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_128 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_348 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_350 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_130 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_352 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_354 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_132 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_356 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_358 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_134 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_360 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_135 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_362 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_136 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_137 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_366 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_42 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_43 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_44 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_149 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_150 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_151 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_29 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_382 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_141 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_384 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_386 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_143 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_388 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_390 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_145 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_392 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_394 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_147 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_396 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_398 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_149 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_400 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_150 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_402 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_151 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_152 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_406 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_39 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_40 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_41 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_142 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_143 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_144 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_27 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_422 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_156 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_424 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_426 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_158 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_428 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_430 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_160 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_432 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_434 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_162 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_436 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_438 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_164 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_440 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_165 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_442 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_166 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_167 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_446 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_36 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_37 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_38 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_135 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_136 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_137 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_25 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_462 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_171 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_464 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_466 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_173 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_468 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_470 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_175 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_472 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_474 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_177 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_476 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_478 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_179 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_480 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_180 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_482 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_181 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_182 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_486 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_33 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_34 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_35 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_128 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_129 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_130 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_23 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_502 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_186 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_504 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_506 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_188 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_508 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_510 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_190 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_512 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_514 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_192 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_516 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_518 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_194 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_520 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_195 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_522 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_196 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_197 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_526 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_30 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_31 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_32 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_121 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_122 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_123 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_21 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_542 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_201 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_544 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_546 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_203 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_548 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_550 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_205 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_552 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_554 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_207 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_556 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_558 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_209 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_560 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_210 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_562 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_211 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_212 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_566 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_27 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_28 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_29 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_114 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_115 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_116 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_19 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_582 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_216 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_584 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_586 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_218 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_588 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_590 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_220 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_592 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_594 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_222 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_596 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_598 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_224 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_600 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_225 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_602 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_226 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_227 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_606 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_24 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_25 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_26 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_107 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_108 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_109 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_17 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_622 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_231 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_624 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_626 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_233 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_628 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_630 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_235 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_632 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_634 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_237 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_636 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_638 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_239 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_640 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_240 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_642 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_241 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_242 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_646 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_21 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_22 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_23 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_100 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_101 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_102 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_15 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_662 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_246 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_664 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_666 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_248 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_668 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_670 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_250 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_672 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_674 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_252 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_676 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_678 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_254 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_680 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_255 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_682 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_256 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_257 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_686 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_18 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_19 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_20 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_93 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_94 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_95 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_13 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_702 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_261 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_704 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_706 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_263 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_708 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_710 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_265 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_712 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_714 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_267 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_716 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_718 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_269 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_720 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_270 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_722 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_271 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_272 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_726 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_15 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_16 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_17 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_86 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_87 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_88 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_11 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_742 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_276 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_744 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_746 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_278 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_748 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_750 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_280 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_752 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_754 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_282 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_756 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_758 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_284 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_760 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_285 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_762 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_286 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_287 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_766 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_12 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_13 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_14 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_79 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_80 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_81 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_9 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_782 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_291 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_784 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_786 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_293 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_788 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_790 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_295 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_792 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_794 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_297 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_796 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_798 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_299 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_800 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_300 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_802 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_301 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_302 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_806 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_9 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_10 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_11 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_72 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_73 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_74 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_7 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_822 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_306 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_824 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_826 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_308 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_828 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_830 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_310 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_832 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_834 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_312 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_836 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_838 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_314 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_840 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_315 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_842 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_316 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_317 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_846 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_6 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_7 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_8 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_65 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_66 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_67 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_5 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_862 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_321 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_864 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_866 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_323 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_868 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_870 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_325 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_872 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_874 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_327 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_876 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_878 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_329 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_880 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_330 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_882 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_331 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_332 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_886 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default_3 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_4 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_5 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_58 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_59 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_60 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_3 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_902 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_336 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_904 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_906 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_338 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_908 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_910 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_340 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_912 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_914 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_342 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_916 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_918 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_344 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_920 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_345 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_922 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_346 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_347 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_926 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_default = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_1 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    clone_default_2 = rand_strided((1, 4, 128, 32), (16384, 4096, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((1, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_52 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_53 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_1 = rand_strided((1, 4, 128, 32), (16384, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    view_942 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_351 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_944 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_946 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_353 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_948 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_950 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_355 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_952 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_954 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_357 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_956 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_958 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_359 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_960 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_360 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_962 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_361 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_49 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sub_27 = rand_strided((128, 30522), (30522, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_483 = rand_strided((512, 128), (1, 512), device='cuda:0', dtype=torch.float32)
    permute_484 = rand_strided((30522, 512), (1, 30522), device='cuda:0', dtype=torch.float32)
    permute_486 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_490 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_494 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_1 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_498 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_502 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_2 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_506 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_510 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_3 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_514 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_518 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_4 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_522 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_526 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_539 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_543 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_547 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_551 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_555 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_559 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_563 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_5 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_567 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_571 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_6 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_575 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_579 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_7 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_583 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_587 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_8 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_591 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_595 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_608 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_612 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_616 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_620 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_624 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_628 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_632 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_9 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_636 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_640 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_10 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_644 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_648 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_11 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_652 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_656 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_12 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_660 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_664 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_677 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_681 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_685 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_689 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_693 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_697 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_701 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_13 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_705 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_709 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_14 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_713 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_717 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_15 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_721 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_725 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_16 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_729 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_733 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_746 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_750 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_754 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_758 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_762 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_766 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_770 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_17 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_774 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_778 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_18 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_782 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_786 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_19 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_790 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_794 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_20 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_798 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_802 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_815 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_819 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_823 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_827 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_831 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_835 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_839 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_21 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_843 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_847 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_22 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_851 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_855 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_23 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_859 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_863 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_24 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_867 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_871 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_884 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_888 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_892 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_896 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_900 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_904 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_908 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_25 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_912 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_916 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_26 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_920 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_924 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_27 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_928 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_932 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_28 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_936 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_940 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_953 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_957 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_961 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_965 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_969 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_973 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_977 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_29 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_981 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_985 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_30 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_989 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_993 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_31 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_997 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1001 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_32 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1005 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1009 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1022 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1026 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1030 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1034 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1038 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1042 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1046 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_33 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1050 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1054 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_34 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1058 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1062 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_35 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1066 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1070 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_36 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1074 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1078 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1091 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1095 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1099 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1103 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1107 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1111 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1115 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_37 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1119 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1123 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_38 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1127 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1131 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_39 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1135 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1139 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_40 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1143 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1147 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1160 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1164 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1168 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1172 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1176 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1180 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1184 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_41 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1188 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1192 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_42 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1196 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1200 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_43 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1204 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1208 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_44 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1212 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1216 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1229 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1233 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1237 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1241 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1245 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1249 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1253 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_45 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1257 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1261 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_46 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1265 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1269 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_47 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1273 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1277 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_48 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1281 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1285 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1298 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1302 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1306 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1310 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1314 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1318 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1322 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_49 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1326 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1330 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_50 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1334 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1338 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_51 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1342 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1346 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_52 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1350 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1354 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1367 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1371 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1375 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1379 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1383 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1387 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1391 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_53 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1395 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1399 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_54 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1403 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1407 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_55 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1411 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1415 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_56 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1419 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1423 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1436 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1440 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1444 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1448 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1452 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1456 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1460 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_57 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1464 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1468 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_58 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1472 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1476 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_59 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1480 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1484 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_60 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1488 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1492 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1505 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1509 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1513 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1517 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1521 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1525 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1529 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_61 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1533 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1537 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_62 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1541 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1545 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_63 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1549 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1553 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_64 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1557 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1561 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1574 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1578 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1582 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1586 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1590 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1594 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1598 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_65 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1602 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1606 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_66 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1610 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1614 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_67 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1618 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1622 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_68 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1626 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1630 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1643 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1647 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1651 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1655 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1659 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1663 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1667 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_69 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1671 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1675 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_70 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1679 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1683 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_71 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1687 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1691 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_72 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1695 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1699 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1712 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1716 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1720 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1724 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1728 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1732 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1736 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_73 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1740 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1744 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_74 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1748 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1752 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_75 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1756 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1760 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_76 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1764 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1768 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1781 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1785 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1789 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1793 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1797 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1801 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1805 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_77 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1809 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1813 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_78 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1817 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1821 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_79 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1825 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1829 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_80 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1833 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1837 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1850 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1854 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1858 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1862 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1866 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1870 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1874 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_81 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1878 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1882 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_82 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1886 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1890 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_83 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1894 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1898 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_84 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1902 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1906 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1919 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1923 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1927 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1931 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1935 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1939 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1943 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_85 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1947 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1951 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_86 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1955 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1959 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_87 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1963 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1967 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_88 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1971 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1975 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1988 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1992 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1996 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2000 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_2004 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_2008 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2012 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_89 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_2016 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2020 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_90 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_2024 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2028 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_91 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_2032 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2036 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_92 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_2040 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2044 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2057 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_2061 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2065 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2069 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_2073 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_2077 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2081 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_93 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_2085 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2089 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_94 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_2093 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2097 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_95 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_2101 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2105 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_96 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_2109 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2113 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2126 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_2130 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2134 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2138 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_2142 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_2146 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 128, 30522), (3906816, 30522, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_21, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_36, primals_37, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_49, primals_50, primals_51, primals_52, primals_53, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_65, primals_66, primals_67, primals_68, primals_69, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_81, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_97, primals_98, primals_99, primals_100, primals_101, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_113, primals_114, primals_115, primals_116, primals_117, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_129, primals_130, primals_131, primals_132, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_145, primals_146, primals_147, primals_148, primals_149, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_161, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_177, primals_178, primals_179, primals_180, primals_181, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_193, primals_194, primals_195, primals_196, primals_197, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_209, primals_210, primals_211, primals_212, primals_213, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_225, primals_226, primals_227, primals_228, primals_229, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_241, primals_242, primals_243, primals_244, primals_245, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_257, primals_258, primals_259, primals_260, primals_261, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_273, primals_274, primals_275, primals_276, primals_277, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_289, primals_290, primals_291, primals_292, primals_293, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_305, primals_306, primals_307, primals_308, primals_309, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_321, primals_322, primals_323, primals_324, primals_325, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_337, primals_338, primals_339, primals_340, primals_341, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_353, primals_354, primals_355, primals_356, primals_357, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_369, primals_370, primals_371, primals_372, primals_373, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_385, primals_1117, primals_1120, primals_1121, full_default, slice_4, view, add_1, view_2, addmm_1, addmm_2, view_6, clone_default_69, clone_default_70, clone_default_71, getitem_212, getitem_213, getitem_214, alias_default_47, view_22, addmm_6, view_24, view_26, addmm_8, view_28, view_30, addmm_10, view_32, view_34, addmm_12, view_36, view_38, addmm_14, view_40, add_16, view_42, addmm_16, addmm_17, view_46, clone_default_66, clone_default_67, clone_default_68, getitem_205, getitem_206, getitem_207, alias_default_45, view_62, addmm_21, view_64, view_66, addmm_23, view_68, view_70, addmm_25, view_72, view_74, addmm_27, view_76, view_78, addmm_29, view_80, addmm_30, view_82, addmm_31, addmm_32, view_86, clone_default_63, clone_default_64, clone_default_65, getitem_198, getitem_199, getitem_200, alias_default_43, view_102, addmm_36, view_104, view_106, addmm_38, view_108, view_110, addmm_40, view_112, view_114, addmm_42, view_116, view_118, addmm_44, view_120, addmm_45, view_122, addmm_46, addmm_47, view_126, clone_default_60, clone_default_61, clone_default_62, getitem_191, getitem_192, getitem_193, alias_default_41, view_142, addmm_51, view_144, view_146, addmm_53, view_148, view_150, addmm_55, view_152, view_154, addmm_57, view_156, view_158, addmm_59, view_160, addmm_60, view_162, addmm_61, addmm_62, view_166, clone_default_57, clone_default_58, clone_default_59, getitem_184, getitem_185, getitem_186, alias_default_39, view_182, addmm_66, view_184, view_186, addmm_68, view_188, view_190, addmm_70, view_192, view_194, addmm_72, view_196, view_198, addmm_74, view_200, addmm_75, view_202, addmm_76, addmm_77, view_206, clone_default_54, clone_default_55, clone_default_56, getitem_177, getitem_178, getitem_179, alias_default_37, view_222, addmm_81, view_224, view_226, addmm_83, view_228, view_230, addmm_85, view_232, view_234, addmm_87, view_236, view_238, addmm_89, view_240, addmm_90, view_242, addmm_91, addmm_92, view_246, clone_default_51, clone_default_52, clone_default_53, getitem_170, getitem_171, getitem_172, alias_default_35, view_262, addmm_96, view_264, view_266, addmm_98, view_268, view_270, addmm_100, view_272, view_274, addmm_102, view_276, view_278, addmm_104, view_280, addmm_105, view_282, addmm_106, addmm_107, view_286, clone_default_48, clone_default_49, clone_default_50, getitem_163, getitem_164, getitem_165, alias_default_33, view_302, addmm_111, view_304, view_306, addmm_113, view_308, view_310, addmm_115, view_312, view_314, addmm_117, view_316, view_318, addmm_119, view_320, addmm_120, view_322, addmm_121, addmm_122, view_326, clone_default_45, clone_default_46, clone_default_47, getitem_156, getitem_157, getitem_158, alias_default_31, view_342, addmm_126, view_344, view_346, addmm_128, view_348, view_350, addmm_130, view_352, view_354, addmm_132, view_356, view_358, addmm_134, view_360, addmm_135, view_362, addmm_136, addmm_137, view_366, clone_default_42, clone_default_43, clone_default_44, getitem_149, getitem_150, getitem_151, alias_default_29, view_382, addmm_141, view_384, view_386, addmm_143, view_388, view_390, addmm_145, view_392, view_394, addmm_147, view_396, view_398, addmm_149, view_400, addmm_150, view_402, addmm_151, addmm_152, view_406, clone_default_39, clone_default_40, clone_default_41, getitem_142, getitem_143, getitem_144, alias_default_27, view_422, addmm_156, view_424, view_426, addmm_158, view_428, view_430, addmm_160, view_432, view_434, addmm_162, view_436, view_438, addmm_164, view_440, addmm_165, view_442, addmm_166, addmm_167, view_446, clone_default_36, clone_default_37, clone_default_38, getitem_135, getitem_136, getitem_137, alias_default_25, view_462, addmm_171, view_464, view_466, addmm_173, view_468, view_470, addmm_175, view_472, view_474, addmm_177, view_476, view_478, addmm_179, view_480, addmm_180, view_482, addmm_181, addmm_182, view_486, clone_default_33, clone_default_34, clone_default_35, getitem_128, getitem_129, getitem_130, alias_default_23, view_502, addmm_186, view_504, view_506, addmm_188, view_508, view_510, addmm_190, view_512, view_514, addmm_192, view_516, view_518, addmm_194, view_520, addmm_195, view_522, addmm_196, addmm_197, view_526, clone_default_30, clone_default_31, clone_default_32, getitem_121, getitem_122, getitem_123, alias_default_21, view_542, addmm_201, view_544, view_546, addmm_203, view_548, view_550, addmm_205, view_552, view_554, addmm_207, view_556, view_558, addmm_209, view_560, addmm_210, view_562, addmm_211, addmm_212, view_566, clone_default_27, clone_default_28, clone_default_29, getitem_114, getitem_115, getitem_116, alias_default_19, view_582, addmm_216, view_584, view_586, addmm_218, view_588, view_590, addmm_220, view_592, view_594, addmm_222, view_596, view_598, addmm_224, view_600, addmm_225, view_602, addmm_226, addmm_227, view_606, clone_default_24, clone_default_25, clone_default_26, getitem_107, getitem_108, getitem_109, alias_default_17, view_622, addmm_231, view_624, view_626, addmm_233, view_628, view_630, addmm_235, view_632, view_634, addmm_237, view_636, view_638, addmm_239, view_640, addmm_240, view_642, addmm_241, addmm_242, view_646, clone_default_21, clone_default_22, clone_default_23, getitem_100, getitem_101, getitem_102, alias_default_15, view_662, addmm_246, view_664, view_666, addmm_248, view_668, view_670, addmm_250, view_672, view_674, addmm_252, view_676, view_678, addmm_254, view_680, addmm_255, view_682, addmm_256, addmm_257, view_686, clone_default_18, clone_default_19, clone_default_20, getitem_93, getitem_94, getitem_95, alias_default_13, view_702, addmm_261, view_704, view_706, addmm_263, view_708, view_710, addmm_265, view_712, view_714, addmm_267, view_716, view_718, addmm_269, view_720, addmm_270, view_722, addmm_271, addmm_272, view_726, clone_default_15, clone_default_16, clone_default_17, getitem_86, getitem_87, getitem_88, alias_default_11, view_742, addmm_276, view_744, view_746, addmm_278, view_748, view_750, addmm_280, view_752, view_754, addmm_282, view_756, view_758, addmm_284, view_760, addmm_285, view_762, addmm_286, addmm_287, view_766, clone_default_12, clone_default_13, clone_default_14, getitem_79, getitem_80, getitem_81, alias_default_9, view_782, addmm_291, view_784, view_786, addmm_293, view_788, view_790, addmm_295, view_792, view_794, addmm_297, view_796, view_798, addmm_299, view_800, addmm_300, view_802, addmm_301, addmm_302, view_806, clone_default_9, clone_default_10, clone_default_11, getitem_72, getitem_73, getitem_74, alias_default_7, view_822, addmm_306, view_824, view_826, addmm_308, view_828, view_830, addmm_310, view_832, view_834, addmm_312, view_836, view_838, addmm_314, view_840, addmm_315, view_842, addmm_316, addmm_317, view_846, clone_default_6, clone_default_7, clone_default_8, getitem_65, getitem_66, getitem_67, alias_default_5, view_862, addmm_321, view_864, view_866, addmm_323, view_868, view_870, addmm_325, view_872, view_874, addmm_327, view_876, view_878, addmm_329, view_880, addmm_330, view_882, addmm_331, addmm_332, view_886, clone_default_3, clone_default_4, clone_default_5, getitem_58, getitem_59, getitem_60, alias_default_3, view_902, addmm_336, view_904, view_906, addmm_338, view_908, view_910, addmm_340, view_912, view_914, addmm_342, view_916, view_918, addmm_344, view_920, addmm_345, view_922, addmm_346, addmm_347, view_926, clone_default, clone_default_1, clone_default_2, getitem_51, getitem_52, getitem_53, alias_default_1, view_942, addmm_351, view_944, view_946, addmm_353, view_948, view_950, addmm_355, view_952, view_954, addmm_357, view_956, view_958, addmm_359, view_960, addmm_360, view_962, addmm_361, getitem_49, rsqrt, sub_27, convert_element_type, permute_483, permute_484, permute_486, permute_490, permute_494, le_1, permute_498, permute_502, le_2, permute_506, permute_510, le_3, permute_514, permute_518, le_4, permute_522, permute_526, permute_539, permute_543, permute_547, permute_551, permute_555, permute_559, permute_563, le_5, permute_567, permute_571, le_6, permute_575, permute_579, le_7, permute_583, permute_587, le_8, permute_591, permute_595, permute_608, permute_612, permute_616, permute_620, permute_624, permute_628, permute_632, le_9, permute_636, permute_640, le_10, permute_644, permute_648, le_11, permute_652, permute_656, le_12, permute_660, permute_664, permute_677, permute_681, permute_685, permute_689, permute_693, permute_697, permute_701, le_13, permute_705, permute_709, le_14, permute_713, permute_717, le_15, permute_721, permute_725, le_16, permute_729, permute_733, permute_746, permute_750, permute_754, permute_758, permute_762, permute_766, permute_770, le_17, permute_774, permute_778, le_18, permute_782, permute_786, le_19, permute_790, permute_794, le_20, permute_798, permute_802, permute_815, permute_819, permute_823, permute_827, permute_831, permute_835, permute_839, le_21, permute_843, permute_847, le_22, permute_851, permute_855, le_23, permute_859, permute_863, le_24, permute_867, permute_871, permute_884, permute_888, permute_892, permute_896, permute_900, permute_904, permute_908, le_25, permute_912, permute_916, le_26, permute_920, permute_924, le_27, permute_928, permute_932, le_28, permute_936, permute_940, permute_953, permute_957, permute_961, permute_965, permute_969, permute_973, permute_977, le_29, permute_981, permute_985, le_30, permute_989, permute_993, le_31, permute_997, permute_1001, le_32, permute_1005, permute_1009, permute_1022, permute_1026, permute_1030, permute_1034, permute_1038, permute_1042, permute_1046, le_33, permute_1050, permute_1054, le_34, permute_1058, permute_1062, le_35, permute_1066, permute_1070, le_36, permute_1074, permute_1078, permute_1091, permute_1095, permute_1099, permute_1103, permute_1107, permute_1111, permute_1115, le_37, permute_1119, permute_1123, le_38, permute_1127, permute_1131, le_39, permute_1135, permute_1139, le_40, permute_1143, permute_1147, permute_1160, permute_1164, permute_1168, permute_1172, permute_1176, permute_1180, permute_1184, le_41, permute_1188, permute_1192, le_42, permute_1196, permute_1200, le_43, permute_1204, permute_1208, le_44, permute_1212, permute_1216, permute_1229, permute_1233, permute_1237, permute_1241, permute_1245, permute_1249, permute_1253, le_45, permute_1257, permute_1261, le_46, permute_1265, permute_1269, le_47, permute_1273, permute_1277, le_48, permute_1281, permute_1285, permute_1298, permute_1302, permute_1306, permute_1310, permute_1314, permute_1318, permute_1322, le_49, permute_1326, permute_1330, le_50, permute_1334, permute_1338, le_51, permute_1342, permute_1346, le_52, permute_1350, permute_1354, permute_1367, permute_1371, permute_1375, permute_1379, permute_1383, permute_1387, permute_1391, le_53, permute_1395, permute_1399, le_54, permute_1403, permute_1407, le_55, permute_1411, permute_1415, le_56, permute_1419, permute_1423, permute_1436, permute_1440, permute_1444, permute_1448, permute_1452, permute_1456, permute_1460, le_57, permute_1464, permute_1468, le_58, permute_1472, permute_1476, le_59, permute_1480, permute_1484, le_60, permute_1488, permute_1492, permute_1505, permute_1509, permute_1513, permute_1517, permute_1521, permute_1525, permute_1529, le_61, permute_1533, permute_1537, le_62, permute_1541, permute_1545, le_63, permute_1549, permute_1553, le_64, permute_1557, permute_1561, permute_1574, permute_1578, permute_1582, permute_1586, permute_1590, permute_1594, permute_1598, le_65, permute_1602, permute_1606, le_66, permute_1610, permute_1614, le_67, permute_1618, permute_1622, le_68, permute_1626, permute_1630, permute_1643, permute_1647, permute_1651, permute_1655, permute_1659, permute_1663, permute_1667, le_69, permute_1671, permute_1675, le_70, permute_1679, permute_1683, le_71, permute_1687, permute_1691, le_72, permute_1695, permute_1699, permute_1712, permute_1716, permute_1720, permute_1724, permute_1728, permute_1732, permute_1736, le_73, permute_1740, permute_1744, le_74, permute_1748, permute_1752, le_75, permute_1756, permute_1760, le_76, permute_1764, permute_1768, permute_1781, permute_1785, permute_1789, permute_1793, permute_1797, permute_1801, permute_1805, le_77, permute_1809, permute_1813, le_78, permute_1817, permute_1821, le_79, permute_1825, permute_1829, le_80, permute_1833, permute_1837, permute_1850, permute_1854, permute_1858, permute_1862, permute_1866, permute_1870, permute_1874, le_81, permute_1878, permute_1882, le_82, permute_1886, permute_1890, le_83, permute_1894, permute_1898, le_84, permute_1902, permute_1906, permute_1919, permute_1923, permute_1927, permute_1931, permute_1935, permute_1939, permute_1943, le_85, permute_1947, permute_1951, le_86, permute_1955, permute_1959, le_87, permute_1963, permute_1967, le_88, permute_1971, permute_1975, permute_1988, permute_1992, permute_1996, permute_2000, permute_2004, permute_2008, permute_2012, le_89, permute_2016, permute_2020, le_90, permute_2024, permute_2028, le_91, permute_2032, permute_2036, le_92, permute_2040, permute_2044, permute_2057, permute_2061, permute_2065, permute_2069, permute_2073, permute_2077, permute_2081, le_93, permute_2085, permute_2089, le_94, permute_2093, permute_2097, le_95, permute_2101, permute_2105, le_96, permute_2109, permute_2113, permute_2126, permute_2130, permute_2134, permute_2138, permute_2142, permute_2146, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MobileBertForMaskedLM', benchmark_compiled_module)
