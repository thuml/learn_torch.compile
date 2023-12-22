
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


# kernel path: /tmp/torchinductor_youkaichao/og/cogs7vyciy563gsp4zyy62aaa3zulgfq7fngowshdp4zefyhlt4j.py
# Source Nodes: [], Original ATen: [aten.nll_loss_backward]

triton_poi_fused_nll_loss_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6e/c6emedne4vc7p7nsrhx4sjvwamm6tmxlqlany2wwtltwah2tkixg.py
# Source Nodes: [end_loss, start_loss], Original ATen: [aten._log_softmax_backward_data, aten.div, aten.nll_loss_backward, aten.nll_loss_forward]
# end_loss => convert_element_type_1, sum_29
# start_loss => convert_element_type, full_default_3, sum_26
triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*i1', 6: '*i1', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (0)).to(tl.int1)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp3 = tl.load(in_ptr2 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp7 = tl.load(in_ptr3 + (0)).to(tl.int1)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp19 = tl.load(in_ptr4 + (r0), rmask, other=0.0)
    tmp20 = tl.load(in_ptr5 + (0)).to(tl.int1)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp22 = tl.load(in_ptr6 + (0)).to(tl.int1)
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp5 = 2.0
    tmp6 = tmp4 / tmp5
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp6 / tmp10
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp24 = tmp23.to(tl.int64)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp6 / tmp25
    tmp27 = tl.where(tmp21, tmp26, tmp12)
    tmp28 = tmp19 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(rmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp18, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp32, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qg/cqgh36ygg6tmlrlo6mj5ktcvbrudw6d6jutqjwcl35xb5chcrltf.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*i1', 10: '*i1', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x1 = (xindex // 2)
    x2 = xindex
    tmp7 = tl.load(in_ptr2 + (0)).to(tl.int1)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp13 = tl.load(in_ptr4 + (0)).to(tl.int1)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp23 = tl.load(in_ptr6 + (0))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp35 = tl.load(in_ptr9 + (0)).to(tl.int1)
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp37 = tl.load(in_ptr10 + (0)).to(tl.int1)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK])
    tmp46 = tl.load(in_ptr12 + (0))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = 2.0
    tmp12 = tmp10 / tmp11
    tmp15 = tmp14.to(tl.int64)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp12 / tmp16
    tmp18 = 0.0
    tmp19 = tl.where(tmp8, tmp17, tmp18)
    tmp20 = tmp6 * tmp19
    tmp21 = tl.load(in_ptr5 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.exp(tmp21)
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 - tmp25
    tmp27 = tmp5 + tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp4, tmp27, tmp28)
    tmp30 = tmp0 >= tmp3
    tmp31 = tl.full([1], 2, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tl.load(in_ptr7 + (x1), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr8 + (x1), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp38.to(tl.int64)
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp12 / tmp40
    tmp42 = tl.where(tmp36, tmp41, tmp18)
    tmp43 = tmp34 * tmp42
    tmp44 = tl.load(in_ptr11 + (x1), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.exp(tmp44)
    tmp48 = tmp45 * tmp47
    tmp49 = tmp43 - tmp48
    tmp50 = tmp33 + tmp49
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp30, tmp50, tmp51)
    tmp53 = tl.where(tmp4, tmp29, tmp52)
    tl.store(out_ptr0 + (x2), tmp53, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nq/cnq4lbnkabdhkdq4ubrt5amtfkzdljnofk7wiqmiepgf735vjd6e.py
# Source Nodes: [], Original ATen: [aten.mul]

triton_poi_fused_mul_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_5', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ba/cbapdvwk54pfpxoedbekeapzutbsc7by2nlhywjbu77egijmrrdc.py
# Source Nodes: [], Original ATen: [aten.mul]

triton_poi_fused_mul_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_6', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/bu/cbugg4qpzbyjvi5apxk4bgtpxzid32zyxet27e5bkljvbeml53ro.py
# Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
# start_loss => full_default_3
triton_poi_fused_nll_loss_forward_threshold_backward_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_forward_threshold_backward_7', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/fp/cfp36s4ap3kxj26qffawrqfstg45n4hhsu4nregr24vj62hra52x.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_8', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/nw/cnwoyyrdpxvvpf6gd3ebvhthp62xy7ywdhshaxd6c5jz5edcu4h7.py
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
triton_poi_fused_add_mul_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_9', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/wb/cwbqux55c4aw627udy34uqlc765ol5hvvsxjswn2ydw3ea46ze4x.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2
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
        tmp0 = tl.load(in_ptr0 + (x0 + (2*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4z/c4zcs2i4ngzxmgyyrg2nxnnrk6zbtvai5ehhmpmba3bafghaf7jo.py
# Source Nodes: [add_361, mul_185, value_tensor_23], Original ATen: [aten.add, aten.mul, aten.sum]
# add_361 => add_361
# mul_185 => mul_185
# value_tensor_23 => add_347
triton_per_fused_add_mul_sum_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sum_11', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/7l/c7lc5evxb4ymhi27nqgxkuilwde545gxp6fcpjpqxl4gm3zmassk.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_12', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/yh/cyh2dd6ytfvflz2ios6ml4mnct22blc73htyt2ozpci3hdnaen4s.py
# Source Nodes: [add_359, attention_output_118, mul_191], Original ATen: [aten.add, aten.mul, aten.sum]
# add_359 => add_359
# attention_output_118 => add_358
# mul_191 => mul_191
triton_red_fused_add_mul_sum_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_13', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/re/crevqygxpje3gx6nqthmi76or7ktajw6tsenf7n4akdrt7xc4ei3.py
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
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_14', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/mx/cmx4hlymmarrlttsjwszp23wmzz7v536pa4mmktfy6ki3dtujmmy.py
# Source Nodes: [add_355, attention_output_116, mul_189], Original ATen: [aten.add, aten.mul, aten.sum]
# add_355 => add_355
# attention_output_116 => add_354
# mul_189 => mul_189
triton_red_fused_add_mul_sum_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_15', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/sy/csykubij6h27oslxsvsk6dw436kte53l63aqp6oiprrroxstibzb.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_16', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/64/c645umcptsey2l2zwovfbubnh2f5rpvp5vtz5n752ovcbuw4jsl3.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_17', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/qz/cqzyiq52ratbee6se2dxgff5iwafhl6hetdyt65nj5dwqier54pi.py
# Source Nodes: [add_351, layer_input_119, mul_186], Original ATen: [aten.add, aten.mul, aten.sum]
# add_351 => add_351
# layer_input_119 => add_348
# mul_186 => mul_186
triton_red_fused_add_mul_sum_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_18', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/nk/cnkvpwhgfdxygwqi2yfwl5j2amapbwe2xtrl5zmyjg5zpmaiiu4u.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_19', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/sy/csywuzxwefn6fg6xd6khpfmmdirxu65rqvwjb2dbo24cah77ywmd.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]

triton_per_fused_add_mul_sum_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sum_20', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/nc/cnc7zewdrvtfgyxtjsw5hfbwap5ek7qxeqpv4brdi4jysj4axwhq.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]

triton_per_fused_add_mul_sum_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sum_22', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/bv/cbvqeq6sjqnldfagsl6y5g3mfnwbbcfa2i7ed45sy3sp6jwznvph.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_23', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/ou/couqatn5wu5vm3mg7rmyb2ia6ahfodcfrxrzmob447ew35peqynm.py
# Source Nodes: [add_331, mul_169, value_tensor_21], Original ATen: [aten.add, aten.mul, aten.sum]
# add_331 => add_331
# mul_169 => mul_169
# value_tensor_21 => add_317
triton_per_fused_add_mul_sum_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_sum_24', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/2y/c2yygjmgqvyg52d2ihreggudvlx4omjdpf6spsg3cut24ldkcaen.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_25', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/ck/cckei47liy636ugq2uprr5ixig3illh6cq6xeyltxu63iop32ddv.py
# Source Nodes: [start_loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.mul, aten.nll_loss_forward]
# start_loss => full_default_3
triton_poi_fused_add_embedding_dense_backward_mul_nll_loss_forward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_dense_backward_mul_nll_loss_forward_26', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/2d/c2d64xkdzxc4rfuq7rmw3tvacgyf46elrqatv5iiqvtkor342qdz.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_27', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/r3/cr3nyz3dvkd5wvj5lftocf6kmlhoeg36emwu45bqp32erxpw44x7.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_28', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/2t/c2t77pd6a6a6afgvepifbkxtpsvo5jmsrungptngksyw67hp2yty.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_29', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/xl/cxlo2bhrbx3jdwc2pj43b42b7aspm24hkt6ulvyzr3thdeds6dqh.py
# Source Nodes: [start_loss], Original ATen: [aten.add, aten.constant_pad_nd, aten.embedding_dense_backward, aten.nll_loss_forward, aten.slice_backward]
# start_loss => full_default_3
triton_poi_fused_add_constant_pad_nd_embedding_dense_backward_nll_loss_forward_slice_backward_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_embedding_dense_backward_nll_loss_forward_slice_backward_30', 'mutated_arg_names': []},
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
    primals_1, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_21, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_36, primals_37, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_49, primals_50, primals_51, primals_52, primals_53, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_65, primals_66, primals_67, primals_68, primals_69, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_81, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_97, primals_98, primals_99, primals_100, primals_101, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_113, primals_114, primals_115, primals_116, primals_117, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_129, primals_130, primals_131, primals_132, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_145, primals_146, primals_147, primals_148, primals_149, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_161, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_177, primals_178, primals_179, primals_180, primals_181, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_193, primals_194, primals_195, primals_196, primals_197, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_209, primals_210, primals_211, primals_212, primals_213, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_225, primals_226, primals_227, primals_228, primals_229, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_241, primals_242, primals_243, primals_244, primals_245, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_257, primals_258, primals_259, primals_260, primals_261, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_273, primals_274, primals_275, primals_276, primals_277, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_289, primals_290, primals_291, primals_292, primals_293, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_305, primals_306, primals_307, primals_308, primals_309, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_321, primals_322, primals_323, primals_324, primals_325, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_337, primals_338, primals_339, primals_340, primals_341, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_353, primals_354, primals_355, primals_356, primals_357, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_369, primals_370, primals_371, primals_372, primals_373, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_385, primals_1115, full_default, slice_4, view, add_1, view_2, addmm_1, addmm_2, view_6, clone_default_69, clone_default_70, clone_default_71, getitem_212, getitem_213, getitem_214, alias_default_47, view_22, addmm_6, view_24, view_26, addmm_8, view_28, view_30, addmm_10, view_32, view_34, addmm_12, view_36, view_38, addmm_14, view_40, add_16, view_42, addmm_16, addmm_17, view_46, clone_default_66, clone_default_67, clone_default_68, getitem_205, getitem_206, getitem_207, alias_default_45, view_62, addmm_21, view_64, view_66, addmm_23, view_68, view_70, addmm_25, view_72, view_74, addmm_27, view_76, view_78, addmm_29, view_80, addmm_30, view_82, addmm_31, addmm_32, view_86, clone_default_63, clone_default_64, clone_default_65, getitem_198, getitem_199, getitem_200, alias_default_43, view_102, addmm_36, view_104, view_106, addmm_38, view_108, view_110, addmm_40, view_112, view_114, addmm_42, view_116, view_118, addmm_44, view_120, addmm_45, view_122, addmm_46, addmm_47, view_126, clone_default_60, clone_default_61, clone_default_62, getitem_191, getitem_192, getitem_193, alias_default_41, view_142, addmm_51, view_144, view_146, addmm_53, view_148, view_150, addmm_55, view_152, view_154, addmm_57, view_156, view_158, addmm_59, view_160, addmm_60, view_162, addmm_61, addmm_62, view_166, clone_default_57, clone_default_58, clone_default_59, getitem_184, getitem_185, getitem_186, alias_default_39, view_182, addmm_66, view_184, view_186, addmm_68, view_188, view_190, addmm_70, view_192, view_194, addmm_72, view_196, view_198, addmm_74, view_200, addmm_75, view_202, addmm_76, addmm_77, view_206, clone_default_54, clone_default_55, clone_default_56, getitem_177, getitem_178, getitem_179, alias_default_37, view_222, addmm_81, view_224, view_226, addmm_83, view_228, view_230, addmm_85, view_232, view_234, addmm_87, view_236, view_238, addmm_89, view_240, addmm_90, view_242, addmm_91, addmm_92, view_246, clone_default_51, clone_default_52, clone_default_53, getitem_170, getitem_171, getitem_172, alias_default_35, view_262, addmm_96, view_264, view_266, addmm_98, view_268, view_270, addmm_100, view_272, view_274, addmm_102, view_276, view_278, addmm_104, view_280, addmm_105, view_282, addmm_106, addmm_107, view_286, clone_default_48, clone_default_49, clone_default_50, getitem_163, getitem_164, getitem_165, alias_default_33, view_302, addmm_111, view_304, view_306, addmm_113, view_308, view_310, addmm_115, view_312, view_314, addmm_117, view_316, view_318, addmm_119, view_320, addmm_120, view_322, addmm_121, addmm_122, view_326, clone_default_45, clone_default_46, clone_default_47, getitem_156, getitem_157, getitem_158, alias_default_31, view_342, addmm_126, view_344, view_346, addmm_128, view_348, view_350, addmm_130, view_352, view_354, addmm_132, view_356, view_358, addmm_134, view_360, addmm_135, view_362, addmm_136, addmm_137, view_366, clone_default_42, clone_default_43, clone_default_44, getitem_149, getitem_150, getitem_151, alias_default_29, view_382, addmm_141, view_384, view_386, addmm_143, view_388, view_390, addmm_145, view_392, view_394, addmm_147, view_396, view_398, addmm_149, view_400, addmm_150, view_402, addmm_151, addmm_152, view_406, clone_default_39, clone_default_40, clone_default_41, getitem_142, getitem_143, getitem_144, alias_default_27, view_422, addmm_156, view_424, view_426, addmm_158, view_428, view_430, addmm_160, view_432, view_434, addmm_162, view_436, view_438, addmm_164, view_440, addmm_165, view_442, addmm_166, addmm_167, view_446, clone_default_36, clone_default_37, clone_default_38, getitem_135, getitem_136, getitem_137, alias_default_25, view_462, addmm_171, view_464, view_466, addmm_173, view_468, view_470, addmm_175, view_472, view_474, addmm_177, view_476, view_478, addmm_179, view_480, addmm_180, view_482, addmm_181, addmm_182, view_486, clone_default_33, clone_default_34, clone_default_35, getitem_128, getitem_129, getitem_130, alias_default_23, view_502, addmm_186, view_504, view_506, addmm_188, view_508, view_510, addmm_190, view_512, view_514, addmm_192, view_516, view_518, addmm_194, view_520, addmm_195, view_522, addmm_196, addmm_197, view_526, clone_default_30, clone_default_31, clone_default_32, getitem_121, getitem_122, getitem_123, alias_default_21, view_542, addmm_201, view_544, view_546, addmm_203, view_548, view_550, addmm_205, view_552, view_554, addmm_207, view_556, view_558, addmm_209, view_560, addmm_210, view_562, addmm_211, addmm_212, view_566, clone_default_27, clone_default_28, clone_default_29, getitem_114, getitem_115, getitem_116, alias_default_19, view_582, addmm_216, view_584, view_586, addmm_218, view_588, view_590, addmm_220, view_592, view_594, addmm_222, view_596, view_598, addmm_224, view_600, addmm_225, view_602, addmm_226, addmm_227, view_606, clone_default_24, clone_default_25, clone_default_26, getitem_107, getitem_108, getitem_109, alias_default_17, view_622, addmm_231, view_624, view_626, addmm_233, view_628, view_630, addmm_235, view_632, view_634, addmm_237, view_636, view_638, addmm_239, view_640, addmm_240, view_642, addmm_241, addmm_242, view_646, clone_default_21, clone_default_22, clone_default_23, getitem_100, getitem_101, getitem_102, alias_default_15, view_662, addmm_246, view_664, view_666, addmm_248, view_668, view_670, addmm_250, view_672, view_674, addmm_252, view_676, view_678, addmm_254, view_680, addmm_255, view_682, addmm_256, addmm_257, view_686, clone_default_18, clone_default_19, clone_default_20, getitem_93, getitem_94, getitem_95, alias_default_13, view_702, addmm_261, view_704, view_706, addmm_263, view_708, view_710, addmm_265, view_712, view_714, addmm_267, view_716, view_718, addmm_269, view_720, addmm_270, view_722, addmm_271, addmm_272, view_726, clone_default_15, clone_default_16, clone_default_17, getitem_86, getitem_87, getitem_88, alias_default_11, view_742, addmm_276, view_744, view_746, addmm_278, view_748, view_750, addmm_280, view_752, view_754, addmm_282, view_756, view_758, addmm_284, view_760, addmm_285, view_762, addmm_286, addmm_287, view_766, clone_default_12, clone_default_13, clone_default_14, getitem_79, getitem_80, getitem_81, alias_default_9, view_782, addmm_291, view_784, view_786, addmm_293, view_788, view_790, addmm_295, view_792, view_794, addmm_297, view_796, view_798, addmm_299, view_800, addmm_300, view_802, addmm_301, addmm_302, view_806, clone_default_9, clone_default_10, clone_default_11, getitem_72, getitem_73, getitem_74, alias_default_7, view_822, addmm_306, view_824, view_826, addmm_308, view_828, view_830, addmm_310, view_832, view_834, addmm_312, view_836, view_838, addmm_314, view_840, addmm_315, view_842, addmm_316, addmm_317, view_846, clone_default_6, clone_default_7, clone_default_8, getitem_65, getitem_66, getitem_67, alias_default_5, view_862, addmm_321, view_864, view_866, addmm_323, view_868, view_870, addmm_325, view_872, view_874, addmm_327, view_876, view_878, addmm_329, view_880, addmm_330, view_882, addmm_331, addmm_332, view_886, clone_default_3, clone_default_4, clone_default_5, getitem_58, getitem_59, getitem_60, alias_default_3, view_902, addmm_336, view_904, view_906, addmm_338, view_908, view_910, addmm_340, view_912, view_914, addmm_342, view_916, view_918, addmm_344, view_920, addmm_345, view_922, addmm_346, addmm_347, view_926, clone_default, clone_default_1, clone_default_2, getitem_51, getitem_52, getitem_53, alias_default_1, view_942, addmm_351, view_944, view_946, addmm_353, view_948, view_950, addmm_355, view_952, view_954, addmm_357, view_956, view_958, addmm_359, view_960, addmm_360, view_962, sub_26, ne, sub_28, ne_3, ne_6, where_4, ne_8, where_6, permute_482, permute_486, permute_490, le, permute_494, permute_498, le_1, permute_502, permute_506, le_2, permute_510, permute_514, le_3, permute_518, permute_522, permute_535, permute_539, permute_543, permute_547, permute_551, permute_555, permute_559, le_4, permute_563, permute_567, le_5, permute_571, permute_575, le_6, permute_579, permute_583, le_7, permute_587, permute_591, permute_604, permute_608, permute_612, permute_616, permute_620, permute_624, permute_628, le_8, permute_632, permute_636, le_9, permute_640, permute_644, le_10, permute_648, permute_652, le_11, permute_656, permute_660, permute_673, permute_677, permute_681, permute_685, permute_689, permute_693, permute_697, le_12, permute_701, permute_705, le_13, permute_709, permute_713, le_14, permute_717, permute_721, le_15, permute_725, permute_729, permute_742, permute_746, permute_750, permute_754, permute_758, permute_762, permute_766, le_16, permute_770, permute_774, le_17, permute_778, permute_782, le_18, permute_786, permute_790, le_19, permute_794, permute_798, permute_811, permute_815, permute_819, permute_823, permute_827, permute_831, permute_835, le_20, permute_839, permute_843, le_21, permute_847, permute_851, le_22, permute_855, permute_859, le_23, permute_863, permute_867, permute_880, permute_884, permute_888, permute_892, permute_896, permute_900, permute_904, le_24, permute_908, permute_912, le_25, permute_916, permute_920, le_26, permute_924, permute_928, le_27, permute_932, permute_936, permute_949, permute_953, permute_957, permute_961, permute_965, permute_969, permute_973, le_28, permute_977, permute_981, le_29, permute_985, permute_989, le_30, permute_993, permute_997, le_31, permute_1001, permute_1005, permute_1018, permute_1022, permute_1026, permute_1030, permute_1034, permute_1038, permute_1042, le_32, permute_1046, permute_1050, le_33, permute_1054, permute_1058, le_34, permute_1062, permute_1066, le_35, permute_1070, permute_1074, permute_1087, permute_1091, permute_1095, permute_1099, permute_1103, permute_1107, permute_1111, le_36, permute_1115, permute_1119, le_37, permute_1123, permute_1127, le_38, permute_1131, permute_1135, le_39, permute_1139, permute_1143, permute_1156, permute_1160, permute_1164, permute_1168, permute_1172, permute_1176, permute_1180, le_40, permute_1184, permute_1188, le_41, permute_1192, permute_1196, le_42, permute_1200, permute_1204, le_43, permute_1208, permute_1212, permute_1225, permute_1229, permute_1233, permute_1237, permute_1241, permute_1245, permute_1249, le_44, permute_1253, permute_1257, le_45, permute_1261, permute_1265, le_46, permute_1269, permute_1273, le_47, permute_1277, permute_1281, permute_1294, permute_1298, permute_1302, permute_1306, permute_1310, permute_1314, permute_1318, le_48, permute_1322, permute_1326, le_49, permute_1330, permute_1334, le_50, permute_1338, permute_1342, le_51, permute_1346, permute_1350, permute_1363, permute_1367, permute_1371, permute_1375, permute_1379, permute_1383, permute_1387, le_52, permute_1391, permute_1395, le_53, permute_1399, permute_1403, le_54, permute_1407, permute_1411, le_55, permute_1415, permute_1419, permute_1432, permute_1436, permute_1440, permute_1444, permute_1448, permute_1452, permute_1456, le_56, permute_1460, permute_1464, le_57, permute_1468, permute_1472, le_58, permute_1476, permute_1480, le_59, permute_1484, permute_1488, permute_1501, permute_1505, permute_1509, permute_1513, permute_1517, permute_1521, permute_1525, le_60, permute_1529, permute_1533, le_61, permute_1537, permute_1541, le_62, permute_1545, permute_1549, le_63, permute_1553, permute_1557, permute_1570, permute_1574, permute_1578, permute_1582, permute_1586, permute_1590, permute_1594, le_64, permute_1598, permute_1602, le_65, permute_1606, permute_1610, le_66, permute_1614, permute_1618, le_67, permute_1622, permute_1626, permute_1639, permute_1643, permute_1647, permute_1651, permute_1655, permute_1659, permute_1663, le_68, permute_1667, permute_1671, le_69, permute_1675, permute_1679, le_70, permute_1683, permute_1687, le_71, permute_1691, permute_1695, permute_1708, permute_1712, permute_1716, permute_1720, permute_1724, permute_1728, permute_1732, le_72, permute_1736, permute_1740, le_73, permute_1744, permute_1748, le_74, permute_1752, permute_1756, le_75, permute_1760, permute_1764, permute_1777, permute_1781, permute_1785, permute_1789, permute_1793, permute_1797, permute_1801, le_76, permute_1805, permute_1809, le_77, permute_1813, permute_1817, le_78, permute_1821, permute_1825, le_79, permute_1829, permute_1833, permute_1846, permute_1850, permute_1854, permute_1858, permute_1862, permute_1866, permute_1870, le_80, permute_1874, permute_1878, le_81, permute_1882, permute_1886, le_82, permute_1890, permute_1894, le_83, permute_1898, permute_1902, permute_1915, permute_1919, permute_1923, permute_1927, permute_1931, permute_1935, permute_1939, le_84, permute_1943, permute_1947, le_85, permute_1951, permute_1955, le_86, permute_1959, permute_1963, le_87, permute_1967, permute_1971, permute_1984, permute_1988, permute_1992, permute_1996, permute_2000, permute_2004, permute_2008, le_88, permute_2012, permute_2016, le_89, permute_2020, permute_2024, le_90, permute_2028, permute_2032, le_91, permute_2036, permute_2040, permute_2053, permute_2057, permute_2061, permute_2065, permute_2069, permute_2073, permute_2077, le_92, permute_2081, permute_2085, le_93, permute_2089, permute_2093, le_94, permute_2097, permute_2101, le_95, permute_2105, permute_2109, permute_2122, permute_2126, permute_2130, permute_2134, permute_2138, permute_2142, tangents_1, tangents_2, tangents_3 = args
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
    assert_size_stride(primals_1115, (1, 128), (128, 1))
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
    assert_size_stride(sub_26, (1, 128), (128, 1))
    assert_size_stride(ne, (1, ), (1, ))
    assert_size_stride(sub_28, (1, 128), (128, 1))
    assert_size_stride(ne_3, (1, ), (1, ))
    assert_size_stride(ne_6, (1, 1), (1, 1))
    assert_size_stride(where_4, (1, 1), (1, 1))
    assert_size_stride(ne_8, (1, 1), (1, 1))
    assert_size_stride(where_6, (1, 1), (1, 1))
    assert_size_stride(permute_482, (2, 512), (512, 1))
    assert_size_stride(permute_486, (512, 128), (128, 1))
    assert_size_stride(permute_490, (128, 512), (512, 1))
    assert_size_stride(le, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_494, (512, 128), (128, 1))
    assert_size_stride(permute_498, (128, 512), (512, 1))
    assert_size_stride(le_1, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_502, (512, 128), (128, 1))
    assert_size_stride(permute_506, (128, 512), (512, 1))
    assert_size_stride(le_2, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_510, (512, 128), (128, 1))
    assert_size_stride(permute_514, (128, 512), (512, 1))
    assert_size_stride(le_3, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_518, (512, 128), (128, 1))
    assert_size_stride(permute_522, (128, 128), (128, 1))
    assert_size_stride(permute_535, (128, 512), (512, 1))
    assert_size_stride(permute_539, (128, 128), (128, 1))
    assert_size_stride(permute_543, (128, 128), (128, 1))
    assert_size_stride(permute_547, (128, 512), (512, 1))
    assert_size_stride(permute_551, (128, 512), (512, 1))
    assert_size_stride(permute_555, (512, 128), (128, 1))
    assert_size_stride(permute_559, (128, 512), (512, 1))
    assert_size_stride(le_4, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_563, (512, 128), (128, 1))
    assert_size_stride(permute_567, (128, 512), (512, 1))
    assert_size_stride(le_5, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_571, (512, 128), (128, 1))
    assert_size_stride(permute_575, (128, 512), (512, 1))
    assert_size_stride(le_6, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_579, (512, 128), (128, 1))
    assert_size_stride(permute_583, (128, 512), (512, 1))
    assert_size_stride(le_7, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_587, (512, 128), (128, 1))
    assert_size_stride(permute_591, (128, 128), (128, 1))
    assert_size_stride(permute_604, (128, 512), (512, 1))
    assert_size_stride(permute_608, (128, 128), (128, 1))
    assert_size_stride(permute_612, (128, 128), (128, 1))
    assert_size_stride(permute_616, (128, 512), (512, 1))
    assert_size_stride(permute_620, (128, 512), (512, 1))
    assert_size_stride(permute_624, (512, 128), (128, 1))
    assert_size_stride(permute_628, (128, 512), (512, 1))
    assert_size_stride(le_8, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_632, (512, 128), (128, 1))
    assert_size_stride(permute_636, (128, 512), (512, 1))
    assert_size_stride(le_9, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_640, (512, 128), (128, 1))
    assert_size_stride(permute_644, (128, 512), (512, 1))
    assert_size_stride(le_10, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_648, (512, 128), (128, 1))
    assert_size_stride(permute_652, (128, 512), (512, 1))
    assert_size_stride(le_11, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_656, (512, 128), (128, 1))
    assert_size_stride(permute_660, (128, 128), (128, 1))
    assert_size_stride(permute_673, (128, 512), (512, 1))
    assert_size_stride(permute_677, (128, 128), (128, 1))
    assert_size_stride(permute_681, (128, 128), (128, 1))
    assert_size_stride(permute_685, (128, 512), (512, 1))
    assert_size_stride(permute_689, (128, 512), (512, 1))
    assert_size_stride(permute_693, (512, 128), (128, 1))
    assert_size_stride(permute_697, (128, 512), (512, 1))
    assert_size_stride(le_12, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_701, (512, 128), (128, 1))
    assert_size_stride(permute_705, (128, 512), (512, 1))
    assert_size_stride(le_13, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_709, (512, 128), (128, 1))
    assert_size_stride(permute_713, (128, 512), (512, 1))
    assert_size_stride(le_14, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_717, (512, 128), (128, 1))
    assert_size_stride(permute_721, (128, 512), (512, 1))
    assert_size_stride(le_15, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_725, (512, 128), (128, 1))
    assert_size_stride(permute_729, (128, 128), (128, 1))
    assert_size_stride(permute_742, (128, 512), (512, 1))
    assert_size_stride(permute_746, (128, 128), (128, 1))
    assert_size_stride(permute_750, (128, 128), (128, 1))
    assert_size_stride(permute_754, (128, 512), (512, 1))
    assert_size_stride(permute_758, (128, 512), (512, 1))
    assert_size_stride(permute_762, (512, 128), (128, 1))
    assert_size_stride(permute_766, (128, 512), (512, 1))
    assert_size_stride(le_16, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_770, (512, 128), (128, 1))
    assert_size_stride(permute_774, (128, 512), (512, 1))
    assert_size_stride(le_17, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_778, (512, 128), (128, 1))
    assert_size_stride(permute_782, (128, 512), (512, 1))
    assert_size_stride(le_18, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_786, (512, 128), (128, 1))
    assert_size_stride(permute_790, (128, 512), (512, 1))
    assert_size_stride(le_19, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_794, (512, 128), (128, 1))
    assert_size_stride(permute_798, (128, 128), (128, 1))
    assert_size_stride(permute_811, (128, 512), (512, 1))
    assert_size_stride(permute_815, (128, 128), (128, 1))
    assert_size_stride(permute_819, (128, 128), (128, 1))
    assert_size_stride(permute_823, (128, 512), (512, 1))
    assert_size_stride(permute_827, (128, 512), (512, 1))
    assert_size_stride(permute_831, (512, 128), (128, 1))
    assert_size_stride(permute_835, (128, 512), (512, 1))
    assert_size_stride(le_20, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_839, (512, 128), (128, 1))
    assert_size_stride(permute_843, (128, 512), (512, 1))
    assert_size_stride(le_21, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_847, (512, 128), (128, 1))
    assert_size_stride(permute_851, (128, 512), (512, 1))
    assert_size_stride(le_22, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_855, (512, 128), (128, 1))
    assert_size_stride(permute_859, (128, 512), (512, 1))
    assert_size_stride(le_23, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_863, (512, 128), (128, 1))
    assert_size_stride(permute_867, (128, 128), (128, 1))
    assert_size_stride(permute_880, (128, 512), (512, 1))
    assert_size_stride(permute_884, (128, 128), (128, 1))
    assert_size_stride(permute_888, (128, 128), (128, 1))
    assert_size_stride(permute_892, (128, 512), (512, 1))
    assert_size_stride(permute_896, (128, 512), (512, 1))
    assert_size_stride(permute_900, (512, 128), (128, 1))
    assert_size_stride(permute_904, (128, 512), (512, 1))
    assert_size_stride(le_24, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_908, (512, 128), (128, 1))
    assert_size_stride(permute_912, (128, 512), (512, 1))
    assert_size_stride(le_25, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_916, (512, 128), (128, 1))
    assert_size_stride(permute_920, (128, 512), (512, 1))
    assert_size_stride(le_26, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_924, (512, 128), (128, 1))
    assert_size_stride(permute_928, (128, 512), (512, 1))
    assert_size_stride(le_27, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_932, (512, 128), (128, 1))
    assert_size_stride(permute_936, (128, 128), (128, 1))
    assert_size_stride(permute_949, (128, 512), (512, 1))
    assert_size_stride(permute_953, (128, 128), (128, 1))
    assert_size_stride(permute_957, (128, 128), (128, 1))
    assert_size_stride(permute_961, (128, 512), (512, 1))
    assert_size_stride(permute_965, (128, 512), (512, 1))
    assert_size_stride(permute_969, (512, 128), (128, 1))
    assert_size_stride(permute_973, (128, 512), (512, 1))
    assert_size_stride(le_28, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_977, (512, 128), (128, 1))
    assert_size_stride(permute_981, (128, 512), (512, 1))
    assert_size_stride(le_29, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_985, (512, 128), (128, 1))
    assert_size_stride(permute_989, (128, 512), (512, 1))
    assert_size_stride(le_30, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_993, (512, 128), (128, 1))
    assert_size_stride(permute_997, (128, 512), (512, 1))
    assert_size_stride(le_31, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1001, (512, 128), (128, 1))
    assert_size_stride(permute_1005, (128, 128), (128, 1))
    assert_size_stride(permute_1018, (128, 512), (512, 1))
    assert_size_stride(permute_1022, (128, 128), (128, 1))
    assert_size_stride(permute_1026, (128, 128), (128, 1))
    assert_size_stride(permute_1030, (128, 512), (512, 1))
    assert_size_stride(permute_1034, (128, 512), (512, 1))
    assert_size_stride(permute_1038, (512, 128), (128, 1))
    assert_size_stride(permute_1042, (128, 512), (512, 1))
    assert_size_stride(le_32, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1046, (512, 128), (128, 1))
    assert_size_stride(permute_1050, (128, 512), (512, 1))
    assert_size_stride(le_33, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1054, (512, 128), (128, 1))
    assert_size_stride(permute_1058, (128, 512), (512, 1))
    assert_size_stride(le_34, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1062, (512, 128), (128, 1))
    assert_size_stride(permute_1066, (128, 512), (512, 1))
    assert_size_stride(le_35, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1070, (512, 128), (128, 1))
    assert_size_stride(permute_1074, (128, 128), (128, 1))
    assert_size_stride(permute_1087, (128, 512), (512, 1))
    assert_size_stride(permute_1091, (128, 128), (128, 1))
    assert_size_stride(permute_1095, (128, 128), (128, 1))
    assert_size_stride(permute_1099, (128, 512), (512, 1))
    assert_size_stride(permute_1103, (128, 512), (512, 1))
    assert_size_stride(permute_1107, (512, 128), (128, 1))
    assert_size_stride(permute_1111, (128, 512), (512, 1))
    assert_size_stride(le_36, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1115, (512, 128), (128, 1))
    assert_size_stride(permute_1119, (128, 512), (512, 1))
    assert_size_stride(le_37, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1123, (512, 128), (128, 1))
    assert_size_stride(permute_1127, (128, 512), (512, 1))
    assert_size_stride(le_38, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1131, (512, 128), (128, 1))
    assert_size_stride(permute_1135, (128, 512), (512, 1))
    assert_size_stride(le_39, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1139, (512, 128), (128, 1))
    assert_size_stride(permute_1143, (128, 128), (128, 1))
    assert_size_stride(permute_1156, (128, 512), (512, 1))
    assert_size_stride(permute_1160, (128, 128), (128, 1))
    assert_size_stride(permute_1164, (128, 128), (128, 1))
    assert_size_stride(permute_1168, (128, 512), (512, 1))
    assert_size_stride(permute_1172, (128, 512), (512, 1))
    assert_size_stride(permute_1176, (512, 128), (128, 1))
    assert_size_stride(permute_1180, (128, 512), (512, 1))
    assert_size_stride(le_40, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1184, (512, 128), (128, 1))
    assert_size_stride(permute_1188, (128, 512), (512, 1))
    assert_size_stride(le_41, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1192, (512, 128), (128, 1))
    assert_size_stride(permute_1196, (128, 512), (512, 1))
    assert_size_stride(le_42, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1200, (512, 128), (128, 1))
    assert_size_stride(permute_1204, (128, 512), (512, 1))
    assert_size_stride(le_43, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1208, (512, 128), (128, 1))
    assert_size_stride(permute_1212, (128, 128), (128, 1))
    assert_size_stride(permute_1225, (128, 512), (512, 1))
    assert_size_stride(permute_1229, (128, 128), (128, 1))
    assert_size_stride(permute_1233, (128, 128), (128, 1))
    assert_size_stride(permute_1237, (128, 512), (512, 1))
    assert_size_stride(permute_1241, (128, 512), (512, 1))
    assert_size_stride(permute_1245, (512, 128), (128, 1))
    assert_size_stride(permute_1249, (128, 512), (512, 1))
    assert_size_stride(le_44, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1253, (512, 128), (128, 1))
    assert_size_stride(permute_1257, (128, 512), (512, 1))
    assert_size_stride(le_45, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1261, (512, 128), (128, 1))
    assert_size_stride(permute_1265, (128, 512), (512, 1))
    assert_size_stride(le_46, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1269, (512, 128), (128, 1))
    assert_size_stride(permute_1273, (128, 512), (512, 1))
    assert_size_stride(le_47, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1277, (512, 128), (128, 1))
    assert_size_stride(permute_1281, (128, 128), (128, 1))
    assert_size_stride(permute_1294, (128, 512), (512, 1))
    assert_size_stride(permute_1298, (128, 128), (128, 1))
    assert_size_stride(permute_1302, (128, 128), (128, 1))
    assert_size_stride(permute_1306, (128, 512), (512, 1))
    assert_size_stride(permute_1310, (128, 512), (512, 1))
    assert_size_stride(permute_1314, (512, 128), (128, 1))
    assert_size_stride(permute_1318, (128, 512), (512, 1))
    assert_size_stride(le_48, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1322, (512, 128), (128, 1))
    assert_size_stride(permute_1326, (128, 512), (512, 1))
    assert_size_stride(le_49, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1330, (512, 128), (128, 1))
    assert_size_stride(permute_1334, (128, 512), (512, 1))
    assert_size_stride(le_50, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1338, (512, 128), (128, 1))
    assert_size_stride(permute_1342, (128, 512), (512, 1))
    assert_size_stride(le_51, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1346, (512, 128), (128, 1))
    assert_size_stride(permute_1350, (128, 128), (128, 1))
    assert_size_stride(permute_1363, (128, 512), (512, 1))
    assert_size_stride(permute_1367, (128, 128), (128, 1))
    assert_size_stride(permute_1371, (128, 128), (128, 1))
    assert_size_stride(permute_1375, (128, 512), (512, 1))
    assert_size_stride(permute_1379, (128, 512), (512, 1))
    assert_size_stride(permute_1383, (512, 128), (128, 1))
    assert_size_stride(permute_1387, (128, 512), (512, 1))
    assert_size_stride(le_52, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1391, (512, 128), (128, 1))
    assert_size_stride(permute_1395, (128, 512), (512, 1))
    assert_size_stride(le_53, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1399, (512, 128), (128, 1))
    assert_size_stride(permute_1403, (128, 512), (512, 1))
    assert_size_stride(le_54, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1407, (512, 128), (128, 1))
    assert_size_stride(permute_1411, (128, 512), (512, 1))
    assert_size_stride(le_55, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1415, (512, 128), (128, 1))
    assert_size_stride(permute_1419, (128, 128), (128, 1))
    assert_size_stride(permute_1432, (128, 512), (512, 1))
    assert_size_stride(permute_1436, (128, 128), (128, 1))
    assert_size_stride(permute_1440, (128, 128), (128, 1))
    assert_size_stride(permute_1444, (128, 512), (512, 1))
    assert_size_stride(permute_1448, (128, 512), (512, 1))
    assert_size_stride(permute_1452, (512, 128), (128, 1))
    assert_size_stride(permute_1456, (128, 512), (512, 1))
    assert_size_stride(le_56, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1460, (512, 128), (128, 1))
    assert_size_stride(permute_1464, (128, 512), (512, 1))
    assert_size_stride(le_57, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1468, (512, 128), (128, 1))
    assert_size_stride(permute_1472, (128, 512), (512, 1))
    assert_size_stride(le_58, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1476, (512, 128), (128, 1))
    assert_size_stride(permute_1480, (128, 512), (512, 1))
    assert_size_stride(le_59, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1484, (512, 128), (128, 1))
    assert_size_stride(permute_1488, (128, 128), (128, 1))
    assert_size_stride(permute_1501, (128, 512), (512, 1))
    assert_size_stride(permute_1505, (128, 128), (128, 1))
    assert_size_stride(permute_1509, (128, 128), (128, 1))
    assert_size_stride(permute_1513, (128, 512), (512, 1))
    assert_size_stride(permute_1517, (128, 512), (512, 1))
    assert_size_stride(permute_1521, (512, 128), (128, 1))
    assert_size_stride(permute_1525, (128, 512), (512, 1))
    assert_size_stride(le_60, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1529, (512, 128), (128, 1))
    assert_size_stride(permute_1533, (128, 512), (512, 1))
    assert_size_stride(le_61, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1537, (512, 128), (128, 1))
    assert_size_stride(permute_1541, (128, 512), (512, 1))
    assert_size_stride(le_62, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1545, (512, 128), (128, 1))
    assert_size_stride(permute_1549, (128, 512), (512, 1))
    assert_size_stride(le_63, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1553, (512, 128), (128, 1))
    assert_size_stride(permute_1557, (128, 128), (128, 1))
    assert_size_stride(permute_1570, (128, 512), (512, 1))
    assert_size_stride(permute_1574, (128, 128), (128, 1))
    assert_size_stride(permute_1578, (128, 128), (128, 1))
    assert_size_stride(permute_1582, (128, 512), (512, 1))
    assert_size_stride(permute_1586, (128, 512), (512, 1))
    assert_size_stride(permute_1590, (512, 128), (128, 1))
    assert_size_stride(permute_1594, (128, 512), (512, 1))
    assert_size_stride(le_64, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1598, (512, 128), (128, 1))
    assert_size_stride(permute_1602, (128, 512), (512, 1))
    assert_size_stride(le_65, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1606, (512, 128), (128, 1))
    assert_size_stride(permute_1610, (128, 512), (512, 1))
    assert_size_stride(le_66, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1614, (512, 128), (128, 1))
    assert_size_stride(permute_1618, (128, 512), (512, 1))
    assert_size_stride(le_67, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1622, (512, 128), (128, 1))
    assert_size_stride(permute_1626, (128, 128), (128, 1))
    assert_size_stride(permute_1639, (128, 512), (512, 1))
    assert_size_stride(permute_1643, (128, 128), (128, 1))
    assert_size_stride(permute_1647, (128, 128), (128, 1))
    assert_size_stride(permute_1651, (128, 512), (512, 1))
    assert_size_stride(permute_1655, (128, 512), (512, 1))
    assert_size_stride(permute_1659, (512, 128), (128, 1))
    assert_size_stride(permute_1663, (128, 512), (512, 1))
    assert_size_stride(le_68, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1667, (512, 128), (128, 1))
    assert_size_stride(permute_1671, (128, 512), (512, 1))
    assert_size_stride(le_69, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1675, (512, 128), (128, 1))
    assert_size_stride(permute_1679, (128, 512), (512, 1))
    assert_size_stride(le_70, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1683, (512, 128), (128, 1))
    assert_size_stride(permute_1687, (128, 512), (512, 1))
    assert_size_stride(le_71, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1691, (512, 128), (128, 1))
    assert_size_stride(permute_1695, (128, 128), (128, 1))
    assert_size_stride(permute_1708, (128, 512), (512, 1))
    assert_size_stride(permute_1712, (128, 128), (128, 1))
    assert_size_stride(permute_1716, (128, 128), (128, 1))
    assert_size_stride(permute_1720, (128, 512), (512, 1))
    assert_size_stride(permute_1724, (128, 512), (512, 1))
    assert_size_stride(permute_1728, (512, 128), (128, 1))
    assert_size_stride(permute_1732, (128, 512), (512, 1))
    assert_size_stride(le_72, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1736, (512, 128), (128, 1))
    assert_size_stride(permute_1740, (128, 512), (512, 1))
    assert_size_stride(le_73, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1744, (512, 128), (128, 1))
    assert_size_stride(permute_1748, (128, 512), (512, 1))
    assert_size_stride(le_74, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1752, (512, 128), (128, 1))
    assert_size_stride(permute_1756, (128, 512), (512, 1))
    assert_size_stride(le_75, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1760, (512, 128), (128, 1))
    assert_size_stride(permute_1764, (128, 128), (128, 1))
    assert_size_stride(permute_1777, (128, 512), (512, 1))
    assert_size_stride(permute_1781, (128, 128), (128, 1))
    assert_size_stride(permute_1785, (128, 128), (128, 1))
    assert_size_stride(permute_1789, (128, 512), (512, 1))
    assert_size_stride(permute_1793, (128, 512), (512, 1))
    assert_size_stride(permute_1797, (512, 128), (128, 1))
    assert_size_stride(permute_1801, (128, 512), (512, 1))
    assert_size_stride(le_76, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1805, (512, 128), (128, 1))
    assert_size_stride(permute_1809, (128, 512), (512, 1))
    assert_size_stride(le_77, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1813, (512, 128), (128, 1))
    assert_size_stride(permute_1817, (128, 512), (512, 1))
    assert_size_stride(le_78, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1821, (512, 128), (128, 1))
    assert_size_stride(permute_1825, (128, 512), (512, 1))
    assert_size_stride(le_79, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1829, (512, 128), (128, 1))
    assert_size_stride(permute_1833, (128, 128), (128, 1))
    assert_size_stride(permute_1846, (128, 512), (512, 1))
    assert_size_stride(permute_1850, (128, 128), (128, 1))
    assert_size_stride(permute_1854, (128, 128), (128, 1))
    assert_size_stride(permute_1858, (128, 512), (512, 1))
    assert_size_stride(permute_1862, (128, 512), (512, 1))
    assert_size_stride(permute_1866, (512, 128), (128, 1))
    assert_size_stride(permute_1870, (128, 512), (512, 1))
    assert_size_stride(le_80, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1874, (512, 128), (128, 1))
    assert_size_stride(permute_1878, (128, 512), (512, 1))
    assert_size_stride(le_81, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1882, (512, 128), (128, 1))
    assert_size_stride(permute_1886, (128, 512), (512, 1))
    assert_size_stride(le_82, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1890, (512, 128), (128, 1))
    assert_size_stride(permute_1894, (128, 512), (512, 1))
    assert_size_stride(le_83, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1898, (512, 128), (128, 1))
    assert_size_stride(permute_1902, (128, 128), (128, 1))
    assert_size_stride(permute_1915, (128, 512), (512, 1))
    assert_size_stride(permute_1919, (128, 128), (128, 1))
    assert_size_stride(permute_1923, (128, 128), (128, 1))
    assert_size_stride(permute_1927, (128, 512), (512, 1))
    assert_size_stride(permute_1931, (128, 512), (512, 1))
    assert_size_stride(permute_1935, (512, 128), (128, 1))
    assert_size_stride(permute_1939, (128, 512), (512, 1))
    assert_size_stride(le_84, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1943, (512, 128), (128, 1))
    assert_size_stride(permute_1947, (128, 512), (512, 1))
    assert_size_stride(le_85, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1951, (512, 128), (128, 1))
    assert_size_stride(permute_1955, (128, 512), (512, 1))
    assert_size_stride(le_86, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1959, (512, 128), (128, 1))
    assert_size_stride(permute_1963, (128, 512), (512, 1))
    assert_size_stride(le_87, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_1967, (512, 128), (128, 1))
    assert_size_stride(permute_1971, (128, 128), (128, 1))
    assert_size_stride(permute_1984, (128, 512), (512, 1))
    assert_size_stride(permute_1988, (128, 128), (128, 1))
    assert_size_stride(permute_1992, (128, 128), (128, 1))
    assert_size_stride(permute_1996, (128, 512), (512, 1))
    assert_size_stride(permute_2000, (128, 512), (512, 1))
    assert_size_stride(permute_2004, (512, 128), (128, 1))
    assert_size_stride(permute_2008, (128, 512), (512, 1))
    assert_size_stride(le_88, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_2012, (512, 128), (128, 1))
    assert_size_stride(permute_2016, (128, 512), (512, 1))
    assert_size_stride(le_89, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_2020, (512, 128), (128, 1))
    assert_size_stride(permute_2024, (128, 512), (512, 1))
    assert_size_stride(le_90, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_2028, (512, 128), (128, 1))
    assert_size_stride(permute_2032, (128, 512), (512, 1))
    assert_size_stride(le_91, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_2036, (512, 128), (128, 1))
    assert_size_stride(permute_2040, (128, 128), (128, 1))
    assert_size_stride(permute_2053, (128, 512), (512, 1))
    assert_size_stride(permute_2057, (128, 128), (128, 1))
    assert_size_stride(permute_2061, (128, 128), (128, 1))
    assert_size_stride(permute_2065, (128, 512), (512, 1))
    assert_size_stride(permute_2069, (128, 512), (512, 1))
    assert_size_stride(permute_2073, (512, 128), (128, 1))
    assert_size_stride(permute_2077, (128, 512), (512, 1))
    assert_size_stride(le_92, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_2081, (512, 128), (128, 1))
    assert_size_stride(permute_2085, (128, 512), (512, 1))
    assert_size_stride(le_93, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_2089, (512, 128), (128, 1))
    assert_size_stride(permute_2093, (128, 512), (512, 1))
    assert_size_stride(le_94, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_2097, (512, 128), (128, 1))
    assert_size_stride(permute_2101, (128, 512), (512, 1))
    assert_size_stride(le_95, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(permute_2105, (512, 128), (128, 1))
    assert_size_stride(permute_2109, (128, 128), (128, 1))
    assert_size_stride(permute_2122, (128, 512), (512, 1))
    assert_size_stride(permute_2126, (128, 128), (128, 1))
    assert_size_stride(permute_2130, (128, 128), (128, 1))
    assert_size_stride(permute_2134, (128, 512), (512, 1))
    assert_size_stride(permute_2138, (128, 512), (512, 1))
    assert_size_stride(permute_2142, (512, 384), (384, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 128), (128, 1))
    assert_size_stride(tangents_3, (1, 128), (128, 1))
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
        buf59 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2.run(buf59, 128, grid=grid(128), stream=stream0)
        aten.scatter_(buf59,1,where_4,-1.0)
        del where_4
        buf63 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2.run(buf63, 128, grid=grid(128), stream=stream0)
        aten.scatter_(buf63,1,where_6,-1.0)
        del where_6
        buf62 = empty((1, 1), device='cuda', dtype=torch.float32)
        buf66 = empty((1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [end_loss, start_loss], Original ATen: [aten._log_softmax_backward_data, aten.div, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_3.run(buf59, ne_6, tangents_1, ne_3, buf63, ne_8, ne, buf62, buf66, 1, 128, grid=grid(1), stream=stream0)
        buf67 = empty((1, 128, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_4.run(tangents_2, buf63, ne_8, tangents_1, ne, sub_26, buf66, tangents_3, buf59, ne_6, ne_3, sub_28, buf62, buf67, 256, grid=grid(256), stream=stream0)
        del buf62
        del buf66
        del ne
        del ne_3
        del ne_6
        del ne_8
        del sub_26
        del sub_28
        del tangents_1
        del tangents_2
        del tangents_3
        buf68 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (128, 2), (2, 1), 0), permute_482, out=buf68)
        del permute_482
        buf73 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf68, primals_385, buf73, 65536, grid=grid(65536), stream=stream0)
        del primals_385
        buf74 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf73, (128, 512), (512, 1), 0), permute_486, out=buf74)
        del permute_486
        buf79 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf74, primals_383, buf79, 16384, grid=grid(16384), stream=stream0)
        del primals_383
        buf80 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (128, 128), (128, 1), 0), permute_490, out=buf80)
        del permute_490
        buf83 = reinterpret_tensor(buf80, (1, 128, 512), (65536, 512, 1), 0); del buf80  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf83, le, 65536, grid=grid(65536), stream=stream0)
        del le
        buf84 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (128, 512), (512, 1), 0), permute_494, out=buf84)
        del permute_494
        buf89 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf79, buf84, primals_381, buf89, 16384, grid=grid(16384), stream=stream0)
        buf90 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf89, (128, 128), (128, 1), 0), permute_498, out=buf90)
        del permute_498
        buf93 = reinterpret_tensor(buf90, (1, 128, 512), (65536, 512, 1), 0); del buf90  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf93, le_1, 65536, grid=grid(65536), stream=stream0)
        del le_1
        buf94 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (128, 512), (512, 1), 0), permute_502, out=buf94)
        del permute_502
        buf57 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf58 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        buf99 = empty((1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_351, add_353, add_355, add_357, attention_output_115, attention_output_116, attention_output_117, layer_input_119, mul_186, mul_188, mul_189, mul_190], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_9.run(addmm_353, addmm_351, addmm_346, primals_371, primals_372, primals_375, primals_376, addmm_357, addmm_355, primals_377, primals_378, primals_379, primals_380, buf89, buf94, buf57, buf58, buf99, 16384, grid=grid(16384), stream=stream0)
        del addmm_353
        del addmm_357
        del primals_376
        del primals_379
        del primals_380
        buf69 = empty((2, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (2, 128), (1, 2), 0), view_962, out=buf69)
        del view_962
        buf70 = empty((1, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf67, buf70, 2, 128, grid=grid(2), stream=stream0)
        del buf67
        buf71 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf72 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_361, mul_185, value_tensor_23], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_11.run(buf68, addmm_360, buf56, primals_369, primals_370, buf71, buf72, 512, 128, grid=grid(512), stream=stream0)
        del addmm_360
        del primals_370
        buf75 = reinterpret_tensor(buf68, (512, 128), (128, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf73, (512, 128), (1, 512), 0), view_960, out=buf75)
        del view_960
        buf76 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf73, buf76, 512, 128, grid=grid(512), stream=stream0)
        buf77 = reinterpret_tensor(buf63, (1, 1, 128), (128, 128, 1), 0); del buf63  # reuse
        buf78 = reinterpret_tensor(buf59, (1, 1, 128), (128, 128, 1), 0); del buf59  # reuse
        buf82 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf87 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf88 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_359, attention_output_118, mul_191], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf74, addmm_359, buf58, primals_381, primals_382, buf79, buf84, buf77, buf78, buf82, buf87, buf88, 128, 128, grid=grid(128), stream=stream0)
        del addmm_359
        del buf58
        del buf74
        del buf84
        del primals_381
        del primals_382
        buf81 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (128, 128), (1, 128), 0), view_958, out=buf81)
        del view_958
        buf85 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (512, 128), (1, 512), 0), view_956, out=buf85)
        del view_956
        buf86 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf83, buf86, 512, 128, grid=grid(512), stream=stream0)
        buf91 = reinterpret_tensor(buf83, (128, 512), (512, 1), 0); del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf89, (128, 128), (1, 128), 0), view_954, out=buf91)
        del view_954
        buf100 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (128, 128), (128, 1), 0), permute_506, out=buf100)
        del permute_506
        buf103 = reinterpret_tensor(buf100, (1, 128, 512), (65536, 512, 1), 0); del buf100  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf103, le_2, 65536, grid=grid(65536), stream=stream0)
        del le_2
        buf104 = reinterpret_tensor(buf79, (128, 128), (128, 1), 0); del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (128, 512), (512, 1), 0), permute_510, out=buf104)
        del permute_510
        buf92 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf97 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf98 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf102 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf107 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf108 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_355, attention_output_116, mul_189], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf89, buf94, addmm_355, buf57, primals_377, primals_378, buf99, buf104, buf92, buf97, buf98, buf102, buf107, buf108, 128, 128, grid=grid(128), stream=stream0)
        del addmm_355
        del buf57
        del primals_378
        buf95 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (512, 128), (1, 512), 0), view_952, out=buf95)
        del view_952
        buf96 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf93, buf96, 512, 128, grid=grid(512), stream=stream0)
        buf101 = reinterpret_tensor(buf93, (128, 512), (512, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (128, 128), (1, 128), 0), view_950, out=buf101)
        del view_950
        buf105 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (512, 128), (1, 512), 0), view_948, out=buf105)
        del view_948
        buf106 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf103, buf106, 512, 128, grid=grid(512), stream=stream0)
        buf109 = reinterpret_tensor(buf104, (1, 128, 128), (16384, 128, 1), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_16.run(buf109, buf99, primals_377, 16384, grid=grid(16384), stream=stream0)
        del primals_377
        buf110 = reinterpret_tensor(buf103, (128, 512), (512, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (128, 128), (128, 1), 0), permute_514, out=buf110)
        del permute_514
        buf111 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (128, 128), (1, 128), 0), view_946, out=buf111)
        del view_946
        buf113 = reinterpret_tensor(buf110, (1, 128, 512), (65536, 512, 1), 0); del buf110  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf113, le_3, 65536, grid=grid(65536), stream=stream0)
        del le_3
        buf114 = reinterpret_tensor(buf99, (128, 128), (128, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (128, 512), (512, 1), 0), permute_518, out=buf114)
        del permute_518
        buf119 = reinterpret_tensor(buf94, (1, 128, 128), (16384, 128, 1), 0); del buf94  # reuse
        buf144 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf109, buf114, primals_375, primals_371, buf119, buf144, 16384, grid=grid(16384), stream=stream0)
        del primals_375
        buf112 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf117 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf118 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf122 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf142 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf143 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_351, layer_input_119, mul_186], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf109, buf114, addmm_351, addmm_346, primals_371, primals_372, buf119, buf112, buf117, buf118, buf122, buf142, buf143, 128, 128, grid=grid(128), stream=stream0)
        del addmm_346
        del addmm_351
        del primals_371
        del primals_372
        buf115 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (512, 128), (1, 512), 0), view_944, out=buf115)
        del view_944
        buf116 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf113, buf116, 512, 128, grid=grid(512), stream=stream0)
        buf120 = buf114; del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (128, 128), (128, 1), 0), permute_522, out=buf120)
        del permute_522
        buf121 = reinterpret_tensor(buf109, (128, 128), (128, 1), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (128, 128), (1, 128), 0), view_942, out=buf121)
        del view_942
        # Source Nodes: [], Original ATen: []
        buf123 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf120, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default, clone_default_1, clone_default_2, None, alias_default_1, getitem_51, getitem_52, getitem_53, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_1
        del clone_default
        del clone_default_1
        del clone_default_2
        del getitem_51
        del getitem_52
        del getitem_53
        buf124 = buf123[0]
        buf125 = buf123[1]
        buf126 = buf123[2]
        del buf123
        buf127 = reinterpret_tensor(buf113, (128, 512), (512, 1), 0); del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf126, (128, 128), (128, 1), 0), permute_535, out=buf127)
        del permute_535
        buf128 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf126, (128, 128), (1, 128), 0), view_922, out=buf128)
        buf129 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf126, buf129, 128, 128, grid=grid(128), stream=stream0)
        buf130 = reinterpret_tensor(buf126, (128, 128), (128, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (128, 128), (128, 1), 0), permute_539, out=buf130)
        del permute_539
        buf131 = buf120; del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (128, 128), (1, 128), 0), view_926, out=buf131)
        buf132 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf125, buf132, 128, 128, grid=grid(128), stream=stream0)
        buf133 = reinterpret_tensor(buf125, (128, 128), (128, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (128, 128), (128, 1), 0), permute_543, out=buf133)
        del permute_543
        buf134 = reinterpret_tensor(buf119, (128, 128), (128, 1), 0); del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (128, 128), (1, 128), 0), view_926, out=buf134)
        del view_926
        buf135 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf124, buf135, 128, 128, grid=grid(128), stream=stream0)
        del buf124
        buf136 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf137 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf130, buf133, addmm_347, buf136, buf137, 128, 128, grid=grid(128), stream=stream0)
        del addmm_347
        buf138 = reinterpret_tensor(buf130, (1, 128, 128), (16384, 128, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf138, buf133, primals_373, 16384, grid=grid(16384), stream=stream0)
        del primals_373
        buf139 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf138, (128, 128), (128, 1), 0), permute_547, out=buf139)
        del permute_547
        buf140 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf138, (128, 128), (1, 128), 0), view_922, out=buf140)
        buf141 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf138, buf141, 128, 128, grid=grid(128), stream=stream0)
        buf145 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (128, 128), (128, 1), 0), permute_551, out=buf145)
        del permute_551
        buf146 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (128, 128), (1, 128), 0), view_922, out=buf146)
        del view_922
        buf147 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf144, buf147, 128, 128, grid=grid(128), stream=stream0)
        buf148 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf150 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_22.run(buf73, buf127, buf139, buf145, buf56, buf148, buf150, 512, 128, grid=grid(512), stream=stream0)
        buf149 = reinterpret_tensor(buf127, (1, 128, 512), (65536, 512, 1), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_23.run(buf149, buf73, buf139, buf145, primals_369, 65536, grid=grid(65536), stream=stream0)
        del primals_369
        buf151 = reinterpret_tensor(buf144, (128, 128), (128, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (128, 512), (512, 1), 0), permute_555, out=buf151)
        del permute_555
        buf152 = reinterpret_tensor(buf73, (512, 128), (128, 1), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (512, 128), (1, 512), 0), view_920, out=buf152)
        del view_920
        buf153 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf149, buf153, 512, 128, grid=grid(512), stream=stream0)
        buf156 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf151, primals_367, buf156, 16384, grid=grid(16384), stream=stream0)
        del primals_367
        buf157 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf156, (128, 128), (128, 1), 0), permute_559, out=buf157)
        del permute_559
        buf160 = reinterpret_tensor(buf157, (1, 128, 512), (65536, 512, 1), 0); del buf157  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf160, le_4, 65536, grid=grid(65536), stream=stream0)
        del le_4
        buf161 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (128, 512), (512, 1), 0), permute_563, out=buf161)
        del permute_563
        buf154 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf155 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf159 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf164 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf165 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_344, attention_output_113, mul_183], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf151, addmm_344, buf55, primals_365, primals_366, buf156, buf161, buf154, buf155, buf159, buf164, buf165, 128, 128, grid=grid(128), stream=stream0)
        del addmm_344
        del primals_366
        buf158 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf156, (128, 128), (1, 128), 0), view_918, out=buf158)
        del view_918
        buf162 = reinterpret_tensor(buf56, (512, 128), (128, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (512, 128), (1, 512), 0), view_916, out=buf162)
        del view_916
        buf163 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf160, buf163, 512, 128, grid=grid(512), stream=stream0)
        buf166 = buf156; del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf166, buf161, primals_365, 16384, grid=grid(16384), stream=stream0)
        del primals_365
        buf167 = reinterpret_tensor(buf160, (128, 512), (512, 1), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (128, 128), (128, 1), 0), permute_567, out=buf167)
        del permute_567
        buf168 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (128, 128), (1, 128), 0), view_914, out=buf168)
        del view_914
        buf170 = reinterpret_tensor(buf167, (1, 128, 512), (65536, 512, 1), 0); del buf167  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf170, le_5, 65536, grid=grid(65536), stream=stream0)
        del le_5
        buf171 = buf161; del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (128, 512), (512, 1), 0), permute_571, out=buf171)
        del permute_571
        buf176 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf166, buf171, primals_363, buf176, 16384, grid=grid(16384), stream=stream0)
        del primals_363
        buf177 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (128, 128), (128, 1), 0), permute_575, out=buf177)
        del permute_575
        buf180 = reinterpret_tensor(buf177, (1, 128, 512), (65536, 512, 1), 0); del buf177  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf180, le_6, 65536, grid=grid(65536), stream=stream0)
        del le_6
        buf181 = buf151; del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf180, (128, 512), (512, 1), 0), permute_579, out=buf181)
        del permute_579
        buf169 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf174 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf175 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf179 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf184 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf185 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_340, attention_output_111, mul_181], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf166, buf171, addmm_340, buf54, primals_361, primals_362, buf176, buf181, buf169, buf174, buf175, buf179, buf184, buf185, 128, 128, grid=grid(128), stream=stream0)
        del addmm_340
        del buf166
        del primals_362
        buf172 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (512, 128), (1, 512), 0), view_912, out=buf172)
        del view_912
        buf173 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf170, buf173, 512, 128, grid=grid(512), stream=stream0)
        buf178 = reinterpret_tensor(buf170, (128, 512), (512, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (128, 128), (1, 128), 0), view_910, out=buf178)
        del view_910
        buf182 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf180, (512, 128), (1, 512), 0), view_908, out=buf182)
        del view_908
        buf183 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf180, buf183, 512, 128, grid=grid(512), stream=stream0)
        buf186 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf186, buf181, primals_361, 16384, grid=grid(16384), stream=stream0)
        del primals_361
        buf187 = reinterpret_tensor(buf180, (128, 512), (512, 1), 0); del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf186, (128, 128), (128, 1), 0), permute_583, out=buf187)
        del permute_583
        buf188 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf186, (128, 128), (1, 128), 0), view_906, out=buf188)
        del view_906
        buf190 = reinterpret_tensor(buf187, (1, 128, 512), (65536, 512, 1), 0); del buf187  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf190, le_7, 65536, grid=grid(65536), stream=stream0)
        del le_7
        buf191 = buf181; del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (128, 512), (512, 1), 0), permute_587, out=buf191)
        del permute_587
        buf196 = buf54; del buf54  # reuse
        buf221 = reinterpret_tensor(buf171, (1, 128, 128), (16384, 128, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf186, buf191, primals_359, primals_355, buf196, buf221, 16384, grid=grid(16384), stream=stream0)
        del primals_359
        buf189 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf194 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf195 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf199 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf219 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf220 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_336, layer_input_114, mul_178], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf186, buf191, addmm_336, addmm_331, primals_355, primals_356, buf196, buf189, buf194, buf195, buf199, buf219, buf220, 128, 128, grid=grid(128), stream=stream0)
        del addmm_331
        del addmm_336
        del primals_355
        del primals_356
        buf192 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (512, 128), (1, 512), 0), view_904, out=buf192)
        del view_904
        buf193 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf190, buf193, 512, 128, grid=grid(512), stream=stream0)
        buf197 = buf191; del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (128, 128), (128, 1), 0), permute_591, out=buf197)
        del permute_591
        buf198 = reinterpret_tensor(buf186, (128, 128), (128, 1), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (128, 128), (1, 128), 0), view_902, out=buf198)
        del view_902
        # Source Nodes: [], Original ATen: []
        buf200 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf197, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_3, clone_default_4, clone_default_5, None, alias_default_3, getitem_58, getitem_59, getitem_60, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_3
        del clone_default_3
        del clone_default_4
        del clone_default_5
        del getitem_58
        del getitem_59
        del getitem_60
        buf201 = buf200[0]
        buf202 = buf200[1]
        buf203 = buf200[2]
        del buf200
        buf204 = reinterpret_tensor(buf190, (128, 512), (512, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf203, (128, 128), (128, 1), 0), permute_604, out=buf204)
        del permute_604
        buf205 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf203, (128, 128), (1, 128), 0), view_882, out=buf205)
        buf206 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf203, buf206, 128, 128, grid=grid(128), stream=stream0)
        buf207 = reinterpret_tensor(buf203, (128, 128), (128, 1), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (128, 128), (128, 1), 0), permute_608, out=buf207)
        del permute_608
        buf208 = buf197; del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (128, 128), (1, 128), 0), view_886, out=buf208)
        buf209 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf202, buf209, 128, 128, grid=grid(128), stream=stream0)
        buf210 = reinterpret_tensor(buf202, (128, 128), (128, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (128, 128), (128, 1), 0), permute_612, out=buf210)
        del permute_612
        buf211 = reinterpret_tensor(buf196, (128, 128), (128, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (128, 128), (1, 128), 0), view_886, out=buf211)
        del view_886
        buf212 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf201, buf212, 128, 128, grid=grid(128), stream=stream0)
        del buf201
        buf213 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf214 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf207, buf210, addmm_332, buf213, buf214, 128, 128, grid=grid(128), stream=stream0)
        del addmm_332
        buf215 = reinterpret_tensor(buf207, (1, 128, 128), (16384, 128, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf215, buf210, primals_357, 16384, grid=grid(16384), stream=stream0)
        del primals_357
        buf216 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (128, 128), (128, 1), 0), permute_616, out=buf216)
        del permute_616
        buf217 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (128, 128), (1, 128), 0), view_882, out=buf217)
        buf218 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf215, buf218, 128, 128, grid=grid(128), stream=stream0)
        buf222 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (128, 128), (128, 1), 0), permute_620, out=buf222)
        del permute_620
        buf223 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (128, 128), (1, 128), 0), view_882, out=buf223)
        del view_882
        buf224 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf221, buf224, 128, 128, grid=grid(128), stream=stream0)
        buf225 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf227 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_331, mul_169, value_tensor_21], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_24.run(buf149, buf204, buf216, buf222, addmm_330, buf51, primals_337, primals_338, buf225, buf227, 512, 128, grid=grid(512), stream=stream0)
        del addmm_330
        del primals_338
        buf226 = buf149; del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf226, buf204, buf216, buf222, primals_353, 65536, grid=grid(65536), stream=stream0)
        del primals_353
        buf228 = reinterpret_tensor(buf221, (128, 128), (128, 1), 0); del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf226, (128, 512), (512, 1), 0), permute_624, out=buf228)
        del permute_624
        buf229 = reinterpret_tensor(buf222, (512, 128), (128, 1), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf226, (512, 128), (1, 512), 0), view_880, out=buf229)
        del view_880
        buf230 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf226, buf230, 512, 128, grid=grid(512), stream=stream0)
        buf233 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf228, primals_351, buf233, 16384, grid=grid(16384), stream=stream0)
        del primals_351
        buf234 = buf216; del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (128, 128), (128, 1), 0), permute_628, out=buf234)
        del permute_628
        buf237 = reinterpret_tensor(buf234, (1, 128, 512), (65536, 512, 1), 0); del buf234  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf237, le_8, 65536, grid=grid(65536), stream=stream0)
        del le_8
        buf238 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (128, 512), (512, 1), 0), permute_632, out=buf238)
        del permute_632
        buf231 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf232 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf236 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf241 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf242 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_329, attention_output_108, mul_175], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf228, addmm_329, buf53, primals_349, primals_350, buf233, buf238, buf231, buf232, buf236, buf241, buf242, 128, 128, grid=grid(128), stream=stream0)
        del addmm_329
        del primals_350
        buf235 = buf204; del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (128, 128), (1, 128), 0), view_878, out=buf235)
        del view_878
        buf239 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (512, 128), (1, 512), 0), view_876, out=buf239)
        del view_876
        buf240 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf237, buf240, 512, 128, grid=grid(512), stream=stream0)
        buf243 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf243, buf238, primals_349, 16384, grid=grid(16384), stream=stream0)
        del primals_349
        buf244 = reinterpret_tensor(buf237, (128, 512), (512, 1), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf243, (128, 128), (128, 1), 0), permute_636, out=buf244)
        del permute_636
        buf245 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf243, (128, 128), (1, 128), 0), view_874, out=buf245)
        del view_874
        buf247 = reinterpret_tensor(buf244, (1, 128, 512), (65536, 512, 1), 0); del buf244  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf247, le_9, 65536, grid=grid(65536), stream=stream0)
        del le_9
        buf248 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf247, (128, 512), (512, 1), 0), permute_640, out=buf248)
        del permute_640
        buf253 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf243, buf248, primals_347, buf253, 16384, grid=grid(16384), stream=stream0)
        del primals_347
        buf254 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf253, (128, 128), (128, 1), 0), permute_644, out=buf254)
        del permute_644
        buf257 = reinterpret_tensor(buf254, (1, 128, 512), (65536, 512, 1), 0); del buf254  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf257, le_10, 65536, grid=grid(65536), stream=stream0)
        del le_10
        buf258 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf257, (128, 512), (512, 1), 0), permute_648, out=buf258)
        del permute_648
        buf246 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf251 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf252 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf256 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf261 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf262 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_325, attention_output_106, mul_173], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf243, buf248, addmm_325, buf52, primals_345, primals_346, buf253, buf258, buf246, buf251, buf252, buf256, buf261, buf262, 128, 128, grid=grid(128), stream=stream0)
        del addmm_325
        del buf243
        del primals_346
        buf249 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf247, (512, 128), (1, 512), 0), view_872, out=buf249)
        del view_872
        buf250 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf247, buf250, 512, 128, grid=grid(512), stream=stream0)
        buf255 = reinterpret_tensor(buf247, (128, 512), (512, 1), 0); del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf253, (128, 128), (1, 128), 0), view_870, out=buf255)
        del view_870
        buf259 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf257, (512, 128), (1, 512), 0), view_868, out=buf259)
        del view_868
        buf260 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf257, buf260, 512, 128, grid=grid(512), stream=stream0)
        buf263 = buf253; del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf263, buf258, primals_345, 16384, grid=grid(16384), stream=stream0)
        del primals_345
        buf264 = reinterpret_tensor(buf257, (128, 512), (512, 1), 0); del buf257  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf263, (128, 128), (128, 1), 0), permute_652, out=buf264)
        del permute_652
        buf265 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf263, (128, 128), (1, 128), 0), view_866, out=buf265)
        del view_866
        buf267 = reinterpret_tensor(buf264, (1, 128, 512), (65536, 512, 1), 0); del buf264  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf267, le_11, 65536, grid=grid(65536), stream=stream0)
        del le_11
        buf268 = buf258; del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf267, (128, 512), (512, 1), 0), permute_656, out=buf268)
        del permute_656
        buf273 = buf52; del buf52  # reuse
        buf298 = reinterpret_tensor(buf248, (1, 128, 128), (16384, 128, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf263, buf268, primals_343, primals_339, buf273, buf298, 16384, grid=grid(16384), stream=stream0)
        del primals_343
        buf266 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf271 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf272 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf276 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf296 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf297 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_321, layer_input_109, mul_170], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf263, buf268, addmm_321, addmm_316, primals_339, primals_340, buf273, buf266, buf271, buf272, buf276, buf296, buf297, 128, 128, grid=grid(128), stream=stream0)
        del addmm_316
        del addmm_321
        del primals_339
        del primals_340
        buf269 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf267, (512, 128), (1, 512), 0), view_864, out=buf269)
        del view_864
        buf270 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf267, buf270, 512, 128, grid=grid(512), stream=stream0)
        buf274 = buf268; del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (128, 128), (128, 1), 0), permute_660, out=buf274)
        del permute_660
        buf275 = reinterpret_tensor(buf263, (128, 128), (128, 1), 0); del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (128, 128), (1, 128), 0), view_862, out=buf275)
        del view_862
        # Source Nodes: [], Original ATen: []
        buf277 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf274, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_6, clone_default_7, clone_default_8, None, alias_default_5, getitem_65, getitem_66, getitem_67, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_5
        del clone_default_6
        del clone_default_7
        del clone_default_8
        del getitem_65
        del getitem_66
        del getitem_67
        buf278 = buf277[0]
        buf279 = buf277[1]
        buf280 = buf277[2]
        del buf277
        buf281 = reinterpret_tensor(buf267, (128, 512), (512, 1), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (128, 128), (128, 1), 0), permute_673, out=buf281)
        del permute_673
        buf282 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (128, 128), (1, 128), 0), view_842, out=buf282)
        buf283 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf280, buf283, 128, 128, grid=grid(128), stream=stream0)
        buf284 = reinterpret_tensor(buf280, (128, 128), (128, 1), 0); del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (128, 128), (128, 1), 0), permute_677, out=buf284)
        del permute_677
        buf285 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (128, 128), (1, 128), 0), view_846, out=buf285)
        buf286 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf279, buf286, 128, 128, grid=grid(128), stream=stream0)
        buf287 = reinterpret_tensor(buf279, (128, 128), (128, 1), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf278, (128, 128), (128, 1), 0), permute_681, out=buf287)
        del permute_681
        buf288 = reinterpret_tensor(buf273, (128, 128), (128, 1), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf278, (128, 128), (1, 128), 0), view_846, out=buf288)
        del view_846
        buf289 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf278, buf289, 128, 128, grid=grid(128), stream=stream0)
        del buf278
        buf290 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf291 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf284, buf287, addmm_317, buf290, buf291, 128, 128, grid=grid(128), stream=stream0)
        del addmm_317
        buf292 = reinterpret_tensor(buf284, (1, 128, 128), (16384, 128, 1), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf292, buf287, primals_341, 16384, grid=grid(16384), stream=stream0)
        del primals_341
        buf293 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf292, (128, 128), (128, 1), 0), permute_685, out=buf293)
        del permute_685
        buf294 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf292, (128, 128), (1, 128), 0), view_842, out=buf294)
        buf295 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf292, buf295, 128, 128, grid=grid(128), stream=stream0)
        buf299 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf298, (128, 128), (128, 1), 0), permute_689, out=buf299)
        del permute_689
        buf300 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf298, (128, 128), (1, 128), 0), view_842, out=buf300)
        del view_842
        buf301 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf298, buf301, 128, 128, grid=grid(128), stream=stream0)
        buf302 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf304 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_22.run(buf226, buf281, buf293, buf299, buf51, buf302, buf304, 512, 128, grid=grid(512), stream=stream0)
        buf303 = buf226; del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf303, buf281, buf293, buf299, primals_337, 65536, grid=grid(65536), stream=stream0)
        del primals_337
        buf305 = reinterpret_tensor(buf298, (128, 128), (128, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf303, (128, 512), (512, 1), 0), permute_693, out=buf305)
        del permute_693
        buf306 = reinterpret_tensor(buf299, (512, 128), (128, 1), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf303, (512, 128), (1, 512), 0), view_840, out=buf306)
        del view_840
        buf307 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf303, buf307, 512, 128, grid=grid(512), stream=stream0)
        buf310 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf305, primals_335, buf310, 16384, grid=grid(16384), stream=stream0)
        del primals_335
        buf311 = buf293; del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (128, 128), (128, 1), 0), permute_697, out=buf311)
        del permute_697
        buf314 = reinterpret_tensor(buf311, (1, 128, 512), (65536, 512, 1), 0); del buf311  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf314, le_12, 65536, grid=grid(65536), stream=stream0)
        del le_12
        buf315 = buf287; del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf314, (128, 512), (512, 1), 0), permute_701, out=buf315)
        del permute_701
        buf308 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf309 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf313 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf318 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf319 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_314, attention_output_103, mul_167], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf305, addmm_314, buf50, primals_333, primals_334, buf310, buf315, buf308, buf309, buf313, buf318, buf319, 128, 128, grid=grid(128), stream=stream0)
        del addmm_314
        del primals_334
        buf312 = buf281; del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (128, 128), (1, 128), 0), view_838, out=buf312)
        del view_838
        buf316 = reinterpret_tensor(buf51, (512, 128), (128, 1), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf314, (512, 128), (1, 512), 0), view_836, out=buf316)
        del view_836
        buf317 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf314, buf317, 512, 128, grid=grid(512), stream=stream0)
        buf320 = buf310; del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf320, buf315, primals_333, 16384, grid=grid(16384), stream=stream0)
        del primals_333
        buf321 = reinterpret_tensor(buf314, (128, 512), (512, 1), 0); del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf320, (128, 128), (128, 1), 0), permute_705, out=buf321)
        del permute_705
        buf322 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf320, (128, 128), (1, 128), 0), view_834, out=buf322)
        del view_834
        buf324 = reinterpret_tensor(buf321, (1, 128, 512), (65536, 512, 1), 0); del buf321  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf324, le_13, 65536, grid=grid(65536), stream=stream0)
        del le_13
        buf325 = buf315; del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf324, (128, 512), (512, 1), 0), permute_709, out=buf325)
        del permute_709
        buf330 = buf50; del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf320, buf325, primals_331, buf330, 16384, grid=grid(16384), stream=stream0)
        del primals_331
        buf331 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf330, (128, 128), (128, 1), 0), permute_713, out=buf331)
        del permute_713
        buf334 = reinterpret_tensor(buf331, (1, 128, 512), (65536, 512, 1), 0); del buf331  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf334, le_14, 65536, grid=grid(65536), stream=stream0)
        del le_14
        buf335 = buf305; del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf334, (128, 512), (512, 1), 0), permute_717, out=buf335)
        del permute_717
        buf323 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf328 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf329 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf333 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf338 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf339 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_310, attention_output_101, mul_165], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf320, buf325, addmm_310, buf49, primals_329, primals_330, buf330, buf335, buf323, buf328, buf329, buf333, buf338, buf339, 128, 128, grid=grid(128), stream=stream0)
        del addmm_310
        del buf320
        del primals_330
        buf326 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf324, (512, 128), (1, 512), 0), view_832, out=buf326)
        del view_832
        buf327 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf324, buf327, 512, 128, grid=grid(512), stream=stream0)
        buf332 = reinterpret_tensor(buf324, (128, 512), (512, 1), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf330, (128, 128), (1, 128), 0), view_830, out=buf332)
        del view_830
        buf336 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf334, (512, 128), (1, 512), 0), view_828, out=buf336)
        del view_828
        buf337 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf334, buf337, 512, 128, grid=grid(512), stream=stream0)
        buf340 = buf330; del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf340, buf335, primals_329, 16384, grid=grid(16384), stream=stream0)
        del primals_329
        buf341 = reinterpret_tensor(buf334, (128, 512), (512, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (128, 128), (128, 1), 0), permute_721, out=buf341)
        del permute_721
        buf342 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (128, 128), (1, 128), 0), view_826, out=buf342)
        del view_826
        buf344 = reinterpret_tensor(buf341, (1, 128, 512), (65536, 512, 1), 0); del buf341  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf344, le_15, 65536, grid=grid(65536), stream=stream0)
        del le_15
        buf345 = buf335; del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf344, (128, 512), (512, 1), 0), permute_725, out=buf345)
        del permute_725
        buf350 = buf49; del buf49  # reuse
        buf375 = reinterpret_tensor(buf325, (1, 128, 128), (16384, 128, 1), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf340, buf345, primals_327, primals_323, buf350, buf375, 16384, grid=grid(16384), stream=stream0)
        del primals_327
        buf343 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf348 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf349 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf353 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf373 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf374 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_306, layer_input_104, mul_162], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf340, buf345, addmm_306, addmm_301, primals_323, primals_324, buf350, buf343, buf348, buf349, buf353, buf373, buf374, 128, 128, grid=grid(128), stream=stream0)
        del addmm_301
        del addmm_306
        del primals_323
        del primals_324
        buf346 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf344, (512, 128), (1, 512), 0), view_824, out=buf346)
        del view_824
        buf347 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf344, buf347, 512, 128, grid=grid(512), stream=stream0)
        buf351 = buf345; del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf350, (128, 128), (128, 1), 0), permute_729, out=buf351)
        del permute_729
        buf352 = reinterpret_tensor(buf340, (128, 128), (128, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf350, (128, 128), (1, 128), 0), view_822, out=buf352)
        del view_822
        # Source Nodes: [], Original ATen: []
        buf354 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf351, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_9, clone_default_10, clone_default_11, None, alias_default_7, getitem_72, getitem_73, getitem_74, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_7
        del clone_default_10
        del clone_default_11
        del clone_default_9
        del getitem_72
        del getitem_73
        del getitem_74
        buf355 = buf354[0]
        buf356 = buf354[1]
        buf357 = buf354[2]
        del buf354
        buf358 = reinterpret_tensor(buf344, (128, 512), (512, 1), 0); del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf357, (128, 128), (128, 1), 0), permute_742, out=buf358)
        del permute_742
        buf359 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf357, (128, 128), (1, 128), 0), view_802, out=buf359)
        buf360 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf357, buf360, 128, 128, grid=grid(128), stream=stream0)
        buf361 = reinterpret_tensor(buf357, (128, 128), (128, 1), 0); del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf356, (128, 128), (128, 1), 0), permute_746, out=buf361)
        del permute_746
        buf362 = buf351; del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf356, (128, 128), (1, 128), 0), view_806, out=buf362)
        buf363 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf356, buf363, 128, 128, grid=grid(128), stream=stream0)
        buf364 = reinterpret_tensor(buf356, (128, 128), (128, 1), 0); del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf355, (128, 128), (128, 1), 0), permute_750, out=buf364)
        del permute_750
        buf365 = reinterpret_tensor(buf350, (128, 128), (128, 1), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf355, (128, 128), (1, 128), 0), view_806, out=buf365)
        del view_806
        buf366 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf355, buf366, 128, 128, grid=grid(128), stream=stream0)
        del buf355
        buf367 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf368 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf361, buf364, addmm_302, buf367, buf368, 128, 128, grid=grid(128), stream=stream0)
        del addmm_302
        buf369 = reinterpret_tensor(buf361, (1, 128, 128), (16384, 128, 1), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf369, buf364, primals_325, 16384, grid=grid(16384), stream=stream0)
        del primals_325
        buf370 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (128, 128), (128, 1), 0), permute_754, out=buf370)
        del permute_754
        buf371 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (128, 128), (1, 128), 0), view_802, out=buf371)
        buf372 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf369, buf372, 128, 128, grid=grid(128), stream=stream0)
        buf376 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (128, 128), (128, 1), 0), permute_758, out=buf376)
        del permute_758
        buf377 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (128, 128), (1, 128), 0), view_802, out=buf377)
        del view_802
        buf378 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf375, buf378, 128, 128, grid=grid(128), stream=stream0)
        buf379 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf381 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_301, mul_153, value_tensor_19], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_24.run(buf303, buf358, buf370, buf376, addmm_300, buf46, primals_305, primals_306, buf379, buf381, 512, 128, grid=grid(512), stream=stream0)
        del addmm_300
        del primals_306
        buf380 = buf303; del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf380, buf358, buf370, buf376, primals_321, 65536, grid=grid(65536), stream=stream0)
        del primals_321
        buf382 = reinterpret_tensor(buf375, (128, 128), (128, 1), 0); del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf380, (128, 512), (512, 1), 0), permute_762, out=buf382)
        del permute_762
        buf383 = reinterpret_tensor(buf376, (512, 128), (128, 1), 0); del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf380, (512, 128), (1, 512), 0), view_800, out=buf383)
        del view_800
        buf384 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf380, buf384, 512, 128, grid=grid(512), stream=stream0)
        buf387 = buf369; del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf382, primals_319, buf387, 16384, grid=grid(16384), stream=stream0)
        del primals_319
        buf388 = buf370; del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (128, 128), (128, 1), 0), permute_766, out=buf388)
        del permute_766
        buf391 = reinterpret_tensor(buf388, (1, 128, 512), (65536, 512, 1), 0); del buf388  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf391, le_16, 65536, grid=grid(65536), stream=stream0)
        del le_16
        buf392 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf391, (128, 512), (512, 1), 0), permute_770, out=buf392)
        del permute_770
        buf385 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf386 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf390 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf395 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf396 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_299, attention_output_98, mul_159], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf382, addmm_299, buf48, primals_317, primals_318, buf387, buf392, buf385, buf386, buf390, buf395, buf396, 128, 128, grid=grid(128), stream=stream0)
        del addmm_299
        del primals_318
        buf389 = buf358; del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (128, 128), (1, 128), 0), view_798, out=buf389)
        del view_798
        buf393 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf391, (512, 128), (1, 512), 0), view_796, out=buf393)
        del view_796
        buf394 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf391, buf394, 512, 128, grid=grid(512), stream=stream0)
        buf397 = buf387; del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf397, buf392, primals_317, 16384, grid=grid(16384), stream=stream0)
        del primals_317
        buf398 = reinterpret_tensor(buf391, (128, 512), (512, 1), 0); del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf397, (128, 128), (128, 1), 0), permute_774, out=buf398)
        del permute_774
        buf399 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf397, (128, 128), (1, 128), 0), view_794, out=buf399)
        del view_794
        buf401 = reinterpret_tensor(buf398, (1, 128, 512), (65536, 512, 1), 0); del buf398  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf401, le_17, 65536, grid=grid(65536), stream=stream0)
        del le_17
        buf402 = buf392; del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf401, (128, 512), (512, 1), 0), permute_778, out=buf402)
        del permute_778
        buf407 = buf48; del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf397, buf402, primals_315, buf407, 16384, grid=grid(16384), stream=stream0)
        del primals_315
        buf408 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf407, (128, 128), (128, 1), 0), permute_782, out=buf408)
        del permute_782
        buf411 = reinterpret_tensor(buf408, (1, 128, 512), (65536, 512, 1), 0); del buf408  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf411, le_18, 65536, grid=grid(65536), stream=stream0)
        del le_18
        buf412 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf411, (128, 512), (512, 1), 0), permute_786, out=buf412)
        del permute_786
        buf400 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf405 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf406 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf410 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf415 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf416 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_295, attention_output_96, mul_157], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf397, buf402, addmm_295, buf47, primals_313, primals_314, buf407, buf412, buf400, buf405, buf406, buf410, buf415, buf416, 128, 128, grid=grid(128), stream=stream0)
        del addmm_295
        del buf397
        del primals_314
        buf403 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf401, (512, 128), (1, 512), 0), view_792, out=buf403)
        del view_792
        buf404 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf401, buf404, 512, 128, grid=grid(512), stream=stream0)
        buf409 = reinterpret_tensor(buf401, (128, 512), (512, 1), 0); del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf407, (128, 128), (1, 128), 0), view_790, out=buf409)
        del view_790
        buf413 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf411, (512, 128), (1, 512), 0), view_788, out=buf413)
        del view_788
        buf414 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf411, buf414, 512, 128, grid=grid(512), stream=stream0)
        buf417 = buf407; del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf417, buf412, primals_313, 16384, grid=grid(16384), stream=stream0)
        del primals_313
        buf418 = reinterpret_tensor(buf411, (128, 512), (512, 1), 0); del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf417, (128, 128), (128, 1), 0), permute_790, out=buf418)
        del permute_790
        buf419 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf417, (128, 128), (1, 128), 0), view_786, out=buf419)
        del view_786
        buf421 = reinterpret_tensor(buf418, (1, 128, 512), (65536, 512, 1), 0); del buf418  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf421, le_19, 65536, grid=grid(65536), stream=stream0)
        del le_19
        buf422 = buf412; del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (128, 512), (512, 1), 0), permute_794, out=buf422)
        del permute_794
        buf427 = buf47; del buf47  # reuse
        buf452 = reinterpret_tensor(buf402, (1, 128, 128), (16384, 128, 1), 0); del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf417, buf422, primals_311, primals_307, buf427, buf452, 16384, grid=grid(16384), stream=stream0)
        del primals_311
        buf420 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf425 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf426 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf430 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf450 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf451 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_291, layer_input_99, mul_154], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf417, buf422, addmm_291, addmm_286, primals_307, primals_308, buf427, buf420, buf425, buf426, buf430, buf450, buf451, 128, 128, grid=grid(128), stream=stream0)
        del addmm_286
        del addmm_291
        del primals_307
        del primals_308
        buf423 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (512, 128), (1, 512), 0), view_784, out=buf423)
        del view_784
        buf424 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf421, buf424, 512, 128, grid=grid(512), stream=stream0)
        buf428 = buf422; del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf427, (128, 128), (128, 1), 0), permute_798, out=buf428)
        del permute_798
        buf429 = reinterpret_tensor(buf417, (128, 128), (128, 1), 0); del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf427, (128, 128), (1, 128), 0), view_782, out=buf429)
        del view_782
        # Source Nodes: [], Original ATen: []
        buf431 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf428, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_12, clone_default_13, clone_default_14, None, alias_default_9, getitem_79, getitem_80, getitem_81, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_9
        del clone_default_12
        del clone_default_13
        del clone_default_14
        del getitem_79
        del getitem_80
        del getitem_81
        buf432 = buf431[0]
        buf433 = buf431[1]
        buf434 = buf431[2]
        del buf431
        buf435 = reinterpret_tensor(buf421, (128, 512), (512, 1), 0); del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf434, (128, 128), (128, 1), 0), permute_811, out=buf435)
        del permute_811
        buf436 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf434, (128, 128), (1, 128), 0), view_762, out=buf436)
        buf437 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf434, buf437, 128, 128, grid=grid(128), stream=stream0)
        buf438 = reinterpret_tensor(buf434, (128, 128), (128, 1), 0); del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf433, (128, 128), (128, 1), 0), permute_815, out=buf438)
        del permute_815
        buf439 = buf428; del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf433, (128, 128), (1, 128), 0), view_766, out=buf439)
        buf440 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf433, buf440, 128, 128, grid=grid(128), stream=stream0)
        buf441 = reinterpret_tensor(buf433, (128, 128), (128, 1), 0); del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf432, (128, 128), (128, 1), 0), permute_819, out=buf441)
        del permute_819
        buf442 = reinterpret_tensor(buf427, (128, 128), (128, 1), 0); del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf432, (128, 128), (1, 128), 0), view_766, out=buf442)
        del view_766
        buf443 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf432, buf443, 128, 128, grid=grid(128), stream=stream0)
        del buf432
        buf444 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf445 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf438, buf441, addmm_287, buf444, buf445, 128, 128, grid=grid(128), stream=stream0)
        del addmm_287
        buf446 = reinterpret_tensor(buf438, (1, 128, 128), (16384, 128, 1), 0); del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf446, buf441, primals_309, 16384, grid=grid(16384), stream=stream0)
        del primals_309
        buf447 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf446, (128, 128), (128, 1), 0), permute_823, out=buf447)
        del permute_823
        buf448 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf446, (128, 128), (1, 128), 0), view_762, out=buf448)
        buf449 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf446, buf449, 128, 128, grid=grid(128), stream=stream0)
        buf453 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf452, (128, 128), (128, 1), 0), permute_827, out=buf453)
        del permute_827
        buf454 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf452, (128, 128), (1, 128), 0), view_762, out=buf454)
        del view_762
        buf455 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf452, buf455, 128, 128, grid=grid(128), stream=stream0)
        buf456 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf458 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_22.run(buf380, buf435, buf447, buf453, buf46, buf456, buf458, 512, 128, grid=grid(512), stream=stream0)
        buf457 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf457, buf435, buf447, buf453, primals_305, 65536, grid=grid(65536), stream=stream0)
        del primals_305
        buf459 = reinterpret_tensor(buf452, (128, 128), (128, 1), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (128, 512), (512, 1), 0), permute_831, out=buf459)
        del permute_831
        buf460 = reinterpret_tensor(buf453, (512, 128), (128, 1), 0); del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (512, 128), (1, 512), 0), view_760, out=buf460)
        del view_760
        buf461 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf457, buf461, 512, 128, grid=grid(512), stream=stream0)
        buf464 = buf446; del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf459, primals_303, buf464, 16384, grid=grid(16384), stream=stream0)
        del primals_303
        buf465 = buf447; del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (128, 128), (128, 1), 0), permute_835, out=buf465)
        del permute_835
        buf468 = reinterpret_tensor(buf465, (1, 128, 512), (65536, 512, 1), 0); del buf465  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf468, le_20, 65536, grid=grid(65536), stream=stream0)
        del le_20
        buf469 = buf441; del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf468, (128, 512), (512, 1), 0), permute_839, out=buf469)
        del permute_839
        buf462 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf463 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf467 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf472 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf473 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_284, attention_output_93, mul_151], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf459, addmm_284, buf45, primals_301, primals_302, buf464, buf469, buf462, buf463, buf467, buf472, buf473, 128, 128, grid=grid(128), stream=stream0)
        del addmm_284
        del primals_302
        buf466 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (128, 128), (1, 128), 0), view_758, out=buf466)
        del view_758
        buf470 = reinterpret_tensor(buf46, (512, 128), (128, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf468, (512, 128), (1, 512), 0), view_756, out=buf470)
        del view_756
        buf471 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf468, buf471, 512, 128, grid=grid(512), stream=stream0)
        buf474 = buf464; del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf474, buf469, primals_301, 16384, grid=grid(16384), stream=stream0)
        del primals_301
        buf475 = reinterpret_tensor(buf468, (128, 512), (512, 1), 0); del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf474, (128, 128), (128, 1), 0), permute_843, out=buf475)
        del permute_843
        buf476 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf474, (128, 128), (1, 128), 0), view_754, out=buf476)
        del view_754
        buf478 = reinterpret_tensor(buf475, (1, 128, 512), (65536, 512, 1), 0); del buf475  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf478, le_21, 65536, grid=grid(65536), stream=stream0)
        del le_21
        buf479 = buf469; del buf469  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf478, (128, 512), (512, 1), 0), permute_847, out=buf479)
        del permute_847
        buf484 = reinterpret_tensor(buf459, (1, 128, 128), (16384, 128, 1), 0); del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf474, buf479, primals_299, buf484, 16384, grid=grid(16384), stream=stream0)
        del primals_299
        buf485 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (128, 128), (128, 1), 0), permute_851, out=buf485)
        del permute_851
        buf488 = reinterpret_tensor(buf485, (1, 128, 512), (65536, 512, 1), 0); del buf485  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf488, le_22, 65536, grid=grid(65536), stream=stream0)
        del le_22
        buf489 = reinterpret_tensor(buf45, (128, 128), (128, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf488, (128, 512), (512, 1), 0), permute_855, out=buf489)
        del permute_855
        buf477 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf482 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf483 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf487 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf492 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf493 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_280, attention_output_91, mul_149], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf474, buf479, addmm_280, buf44, primals_297, primals_298, buf484, buf489, buf477, buf482, buf483, buf487, buf492, buf493, 128, 128, grid=grid(128), stream=stream0)
        del addmm_280
        del buf44
        del primals_298
        buf480 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf478, (512, 128), (1, 512), 0), view_752, out=buf480)
        del view_752
        buf481 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf478, buf481, 512, 128, grid=grid(512), stream=stream0)
        buf486 = reinterpret_tensor(buf478, (128, 512), (512, 1), 0); del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (128, 128), (1, 128), 0), view_750, out=buf486)
        del view_750
        buf490 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf488, (512, 128), (1, 512), 0), view_748, out=buf490)
        del view_748
        buf491 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf488, buf491, 512, 128, grid=grid(512), stream=stream0)
        buf494 = buf484; del buf484  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf494, buf489, primals_297, 16384, grid=grid(16384), stream=stream0)
        del primals_297
        buf495 = reinterpret_tensor(buf488, (128, 512), (512, 1), 0); del buf488  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf494, (128, 128), (128, 1), 0), permute_859, out=buf495)
        del permute_859
        buf496 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf494, (128, 128), (1, 128), 0), view_746, out=buf496)
        del view_746
        buf498 = reinterpret_tensor(buf495, (1, 128, 512), (65536, 512, 1), 0); del buf495  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf498, le_23, 65536, grid=grid(65536), stream=stream0)
        del le_23
        buf499 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf498, (128, 512), (512, 1), 0), permute_863, out=buf499)
        del permute_863
        buf504 = reinterpret_tensor(buf479, (1, 128, 128), (16384, 128, 1), 0); del buf479  # reuse
        buf529 = buf474; del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf494, buf499, primals_295, primals_291, buf504, buf529, 16384, grid=grid(16384), stream=stream0)
        del primals_295
        buf497 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf502 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf503 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf507 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf527 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf528 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_276, layer_input_94, mul_146], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf494, buf499, addmm_276, addmm_271, primals_291, primals_292, buf504, buf497, buf502, buf503, buf507, buf527, buf528, 128, 128, grid=grid(128), stream=stream0)
        del addmm_271
        del addmm_276
        del primals_291
        del primals_292
        buf500 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf498, (512, 128), (1, 512), 0), view_744, out=buf500)
        del view_744
        buf501 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf498, buf501, 512, 128, grid=grid(512), stream=stream0)
        buf505 = buf499; del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf504, (128, 128), (128, 1), 0), permute_867, out=buf505)
        del permute_867
        buf506 = reinterpret_tensor(buf494, (128, 128), (128, 1), 0); del buf494  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf504, (128, 128), (1, 128), 0), view_742, out=buf506)
        del view_742
        # Source Nodes: [], Original ATen: []
        buf508 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf505, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_15, clone_default_16, clone_default_17, None, alias_default_11, getitem_86, getitem_87, getitem_88, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_11
        del clone_default_15
        del clone_default_16
        del clone_default_17
        del getitem_86
        del getitem_87
        del getitem_88
        buf509 = buf508[0]
        buf510 = buf508[1]
        buf511 = buf508[2]
        del buf508
        buf512 = reinterpret_tensor(buf498, (128, 512), (512, 1), 0); del buf498  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf511, (128, 128), (128, 1), 0), permute_880, out=buf512)
        del permute_880
        buf513 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf511, (128, 128), (1, 128), 0), view_722, out=buf513)
        buf514 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf511, buf514, 128, 128, grid=grid(128), stream=stream0)
        buf515 = reinterpret_tensor(buf511, (128, 128), (128, 1), 0); del buf511  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf510, (128, 128), (128, 1), 0), permute_884, out=buf515)
        del permute_884
        buf516 = buf505; del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf510, (128, 128), (1, 128), 0), view_726, out=buf516)
        buf517 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf510, buf517, 128, 128, grid=grid(128), stream=stream0)
        buf518 = reinterpret_tensor(buf510, (128, 128), (128, 1), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (128, 128), (128, 1), 0), permute_888, out=buf518)
        del permute_888
        buf519 = reinterpret_tensor(buf504, (128, 128), (128, 1), 0); del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (128, 128), (1, 128), 0), view_726, out=buf519)
        del view_726
        buf520 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf509, buf520, 128, 128, grid=grid(128), stream=stream0)
        del buf509
        buf521 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf522 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf515, buf518, addmm_272, buf521, buf522, 128, 128, grid=grid(128), stream=stream0)
        del addmm_272
        buf523 = reinterpret_tensor(buf515, (1, 128, 128), (16384, 128, 1), 0); del buf515  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf523, buf518, primals_293, 16384, grid=grid(16384), stream=stream0)
        del primals_293
        buf524 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf523, (128, 128), (128, 1), 0), permute_892, out=buf524)
        del permute_892
        buf525 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf523, (128, 128), (1, 128), 0), view_722, out=buf525)
        buf526 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf523, buf526, 128, 128, grid=grid(128), stream=stream0)
        buf530 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf529, (128, 128), (128, 1), 0), permute_896, out=buf530)
        del permute_896
        buf531 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf529, (128, 128), (1, 128), 0), view_722, out=buf531)
        del view_722
        buf532 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf529, buf532, 128, 128, grid=grid(128), stream=stream0)
        buf533 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf535 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_271, mul_137, value_tensor_17], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_24.run(buf457, buf512, buf524, buf530, addmm_270, buf41, primals_273, primals_274, buf533, buf535, 512, 128, grid=grid(512), stream=stream0)
        del addmm_270
        del primals_274
        buf534 = buf457; del buf457  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf534, buf512, buf524, buf530, primals_289, 65536, grid=grid(65536), stream=stream0)
        del primals_289
        buf536 = reinterpret_tensor(buf529, (128, 128), (128, 1), 0); del buf529  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf534, (128, 512), (512, 1), 0), permute_900, out=buf536)
        del permute_900
        buf537 = reinterpret_tensor(buf530, (512, 128), (128, 1), 0); del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf534, (512, 128), (1, 512), 0), view_720, out=buf537)
        del view_720
        buf538 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf534, buf538, 512, 128, grid=grid(512), stream=stream0)
        buf541 = buf523; del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf536, primals_287, buf541, 16384, grid=grid(16384), stream=stream0)
        del primals_287
        buf542 = buf524; del buf524  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf541, (128, 128), (128, 1), 0), permute_904, out=buf542)
        del permute_904
        buf545 = reinterpret_tensor(buf542, (1, 128, 512), (65536, 512, 1), 0); del buf542  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf545, le_24, 65536, grid=grid(65536), stream=stream0)
        del le_24
        buf546 = buf518; del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf545, (128, 512), (512, 1), 0), permute_908, out=buf546)
        del permute_908
        buf539 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf540 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf544 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf549 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf550 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_269, attention_output_88, mul_143], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf536, addmm_269, buf43, primals_285, primals_286, buf541, buf546, buf539, buf540, buf544, buf549, buf550, 128, 128, grid=grid(128), stream=stream0)
        del addmm_269
        del primals_286
        buf543 = buf512; del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf541, (128, 128), (1, 128), 0), view_718, out=buf543)
        del view_718
        buf547 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf545, (512, 128), (1, 512), 0), view_716, out=buf547)
        del view_716
        buf548 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf545, buf548, 512, 128, grid=grid(512), stream=stream0)
        buf551 = buf541; del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf551, buf546, primals_285, 16384, grid=grid(16384), stream=stream0)
        del primals_285
        buf552 = reinterpret_tensor(buf545, (128, 512), (512, 1), 0); del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf551, (128, 128), (128, 1), 0), permute_912, out=buf552)
        del permute_912
        buf553 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf551, (128, 128), (1, 128), 0), view_714, out=buf553)
        del view_714
        buf555 = reinterpret_tensor(buf552, (1, 128, 512), (65536, 512, 1), 0); del buf552  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf555, le_25, 65536, grid=grid(65536), stream=stream0)
        del le_25
        buf556 = buf546; del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf555, (128, 512), (512, 1), 0), permute_916, out=buf556)
        del permute_916
        buf561 = reinterpret_tensor(buf536, (1, 128, 128), (16384, 128, 1), 0); del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf551, buf556, primals_283, buf561, 16384, grid=grid(16384), stream=stream0)
        del primals_283
        buf562 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf561, (128, 128), (128, 1), 0), permute_920, out=buf562)
        del permute_920
        buf565 = reinterpret_tensor(buf562, (1, 128, 512), (65536, 512, 1), 0); del buf562  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf565, le_26, 65536, grid=grid(65536), stream=stream0)
        del le_26
        buf566 = reinterpret_tensor(buf43, (128, 128), (128, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf565, (128, 512), (512, 1), 0), permute_924, out=buf566)
        del permute_924
        buf554 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf559 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf560 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf564 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf569 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf570 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_265, attention_output_86, mul_141], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf551, buf556, addmm_265, buf42, primals_281, primals_282, buf561, buf566, buf554, buf559, buf560, buf564, buf569, buf570, 128, 128, grid=grid(128), stream=stream0)
        del addmm_265
        del buf42
        del primals_282
        buf557 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf555, (512, 128), (1, 512), 0), view_712, out=buf557)
        del view_712
        buf558 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf555, buf558, 512, 128, grid=grid(512), stream=stream0)
        buf563 = reinterpret_tensor(buf555, (128, 512), (512, 1), 0); del buf555  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf561, (128, 128), (1, 128), 0), view_710, out=buf563)
        del view_710
        buf567 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf565, (512, 128), (1, 512), 0), view_708, out=buf567)
        del view_708
        buf568 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf565, buf568, 512, 128, grid=grid(512), stream=stream0)
        buf571 = buf561; del buf561  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf571, buf566, primals_281, 16384, grid=grid(16384), stream=stream0)
        del primals_281
        buf572 = reinterpret_tensor(buf565, (128, 512), (512, 1), 0); del buf565  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf571, (128, 128), (128, 1), 0), permute_928, out=buf572)
        del permute_928
        buf573 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf571, (128, 128), (1, 128), 0), view_706, out=buf573)
        del view_706
        buf575 = reinterpret_tensor(buf572, (1, 128, 512), (65536, 512, 1), 0); del buf572  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf575, le_27, 65536, grid=grid(65536), stream=stream0)
        del le_27
        buf576 = buf566; del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf575, (128, 512), (512, 1), 0), permute_932, out=buf576)
        del permute_932
        buf581 = reinterpret_tensor(buf556, (1, 128, 128), (16384, 128, 1), 0); del buf556  # reuse
        buf606 = buf551; del buf551  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf571, buf576, primals_279, primals_275, buf581, buf606, 16384, grid=grid(16384), stream=stream0)
        del primals_279
        buf574 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf579 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf580 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf584 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf604 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf605 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_261, layer_input_89, mul_138], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf571, buf576, addmm_261, addmm_256, primals_275, primals_276, buf581, buf574, buf579, buf580, buf584, buf604, buf605, 128, 128, grid=grid(128), stream=stream0)
        del addmm_256
        del addmm_261
        del primals_275
        del primals_276
        buf577 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf575, (512, 128), (1, 512), 0), view_704, out=buf577)
        del view_704
        buf578 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf575, buf578, 512, 128, grid=grid(512), stream=stream0)
        buf582 = buf576; del buf576  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf581, (128, 128), (128, 1), 0), permute_936, out=buf582)
        del permute_936
        buf583 = reinterpret_tensor(buf571, (128, 128), (128, 1), 0); del buf571  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf581, (128, 128), (1, 128), 0), view_702, out=buf583)
        del view_702
        # Source Nodes: [], Original ATen: []
        buf585 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf582, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_18, clone_default_19, clone_default_20, None, alias_default_13, getitem_93, getitem_94, getitem_95, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_13
        del clone_default_18
        del clone_default_19
        del clone_default_20
        del getitem_93
        del getitem_94
        del getitem_95
        buf586 = buf585[0]
        buf587 = buf585[1]
        buf588 = buf585[2]
        del buf585
        buf589 = reinterpret_tensor(buf575, (128, 512), (512, 1), 0); del buf575  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf588, (128, 128), (128, 1), 0), permute_949, out=buf589)
        del permute_949
        buf590 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf588, (128, 128), (1, 128), 0), view_682, out=buf590)
        buf591 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf588, buf591, 128, 128, grid=grid(128), stream=stream0)
        buf592 = reinterpret_tensor(buf588, (128, 128), (128, 1), 0); del buf588  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf587, (128, 128), (128, 1), 0), permute_953, out=buf592)
        del permute_953
        buf593 = buf582; del buf582  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf587, (128, 128), (1, 128), 0), view_686, out=buf593)
        buf594 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf587, buf594, 128, 128, grid=grid(128), stream=stream0)
        buf595 = reinterpret_tensor(buf587, (128, 128), (128, 1), 0); del buf587  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf586, (128, 128), (128, 1), 0), permute_957, out=buf595)
        del permute_957
        buf596 = reinterpret_tensor(buf581, (128, 128), (128, 1), 0); del buf581  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf586, (128, 128), (1, 128), 0), view_686, out=buf596)
        del view_686
        buf597 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf586, buf597, 128, 128, grid=grid(128), stream=stream0)
        del buf586
        buf598 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf599 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf592, buf595, addmm_257, buf598, buf599, 128, 128, grid=grid(128), stream=stream0)
        del addmm_257
        buf600 = reinterpret_tensor(buf592, (1, 128, 128), (16384, 128, 1), 0); del buf592  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf600, buf595, primals_277, 16384, grid=grid(16384), stream=stream0)
        del primals_277
        buf601 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf600, (128, 128), (128, 1), 0), permute_961, out=buf601)
        del permute_961
        buf602 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf600, (128, 128), (1, 128), 0), view_682, out=buf602)
        buf603 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf600, buf603, 128, 128, grid=grid(128), stream=stream0)
        buf607 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf606, (128, 128), (128, 1), 0), permute_965, out=buf607)
        del permute_965
        buf608 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf606, (128, 128), (1, 128), 0), view_682, out=buf608)
        del view_682
        buf609 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf606, buf609, 128, 128, grid=grid(128), stream=stream0)
        buf610 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf612 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_22.run(buf534, buf589, buf601, buf607, buf41, buf610, buf612, 512, 128, grid=grid(512), stream=stream0)
        buf611 = buf534; del buf534  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf611, buf589, buf601, buf607, primals_273, 65536, grid=grid(65536), stream=stream0)
        del primals_273
        buf613 = reinterpret_tensor(buf606, (128, 128), (128, 1), 0); del buf606  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf611, (128, 512), (512, 1), 0), permute_969, out=buf613)
        del permute_969
        buf614 = reinterpret_tensor(buf607, (512, 128), (128, 1), 0); del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf611, (512, 128), (1, 512), 0), view_680, out=buf614)
        del view_680
        buf615 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf611, buf615, 512, 128, grid=grid(512), stream=stream0)
        buf618 = buf600; del buf600  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf613, primals_271, buf618, 16384, grid=grid(16384), stream=stream0)
        del primals_271
        buf619 = buf601; del buf601  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf618, (128, 128), (128, 1), 0), permute_973, out=buf619)
        del permute_973
        buf622 = reinterpret_tensor(buf619, (1, 128, 512), (65536, 512, 1), 0); del buf619  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf622, le_28, 65536, grid=grid(65536), stream=stream0)
        del le_28
        buf623 = buf595; del buf595  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf622, (128, 512), (512, 1), 0), permute_977, out=buf623)
        del permute_977
        buf616 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf617 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf621 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf626 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf627 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_254, attention_output_83, mul_135], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf613, addmm_254, buf40, primals_269, primals_270, buf618, buf623, buf616, buf617, buf621, buf626, buf627, 128, 128, grid=grid(128), stream=stream0)
        del addmm_254
        del primals_270
        buf620 = buf589; del buf589  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf618, (128, 128), (1, 128), 0), view_678, out=buf620)
        del view_678
        buf624 = reinterpret_tensor(buf41, (512, 128), (128, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf622, (512, 128), (1, 512), 0), view_676, out=buf624)
        del view_676
        buf625 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf622, buf625, 512, 128, grid=grid(512), stream=stream0)
        buf628 = buf618; del buf618  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf628, buf623, primals_269, 16384, grid=grid(16384), stream=stream0)
        del primals_269
        buf629 = reinterpret_tensor(buf622, (128, 512), (512, 1), 0); del buf622  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf628, (128, 128), (128, 1), 0), permute_981, out=buf629)
        del permute_981
        buf630 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf628, (128, 128), (1, 128), 0), view_674, out=buf630)
        del view_674
        buf632 = reinterpret_tensor(buf629, (1, 128, 512), (65536, 512, 1), 0); del buf629  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf632, le_29, 65536, grid=grid(65536), stream=stream0)
        del le_29
        buf633 = buf623; del buf623  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf632, (128, 512), (512, 1), 0), permute_985, out=buf633)
        del permute_985
        buf638 = reinterpret_tensor(buf613, (1, 128, 128), (16384, 128, 1), 0); del buf613  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf628, buf633, primals_267, buf638, 16384, grid=grid(16384), stream=stream0)
        del primals_267
        buf639 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf638, (128, 128), (128, 1), 0), permute_989, out=buf639)
        del permute_989
        buf642 = reinterpret_tensor(buf639, (1, 128, 512), (65536, 512, 1), 0); del buf639  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf642, le_30, 65536, grid=grid(65536), stream=stream0)
        del le_30
        buf643 = reinterpret_tensor(buf40, (128, 128), (128, 1), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf642, (128, 512), (512, 1), 0), permute_993, out=buf643)
        del permute_993
        buf631 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf636 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf637 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf641 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf646 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf647 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_250, attention_output_81, mul_133], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf628, buf633, addmm_250, buf39, primals_265, primals_266, buf638, buf643, buf631, buf636, buf637, buf641, buf646, buf647, 128, 128, grid=grid(128), stream=stream0)
        del addmm_250
        del buf39
        del primals_266
        buf634 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf632, (512, 128), (1, 512), 0), view_672, out=buf634)
        del view_672
        buf635 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf632, buf635, 512, 128, grid=grid(512), stream=stream0)
        buf640 = reinterpret_tensor(buf632, (128, 512), (512, 1), 0); del buf632  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf638, (128, 128), (1, 128), 0), view_670, out=buf640)
        del view_670
        buf644 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf642, (512, 128), (1, 512), 0), view_668, out=buf644)
        del view_668
        buf645 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf642, buf645, 512, 128, grid=grid(512), stream=stream0)
        buf648 = buf638; del buf638  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf648, buf643, primals_265, 16384, grid=grid(16384), stream=stream0)
        del primals_265
        buf649 = reinterpret_tensor(buf642, (128, 512), (512, 1), 0); del buf642  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf648, (128, 128), (128, 1), 0), permute_997, out=buf649)
        del permute_997
        buf650 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf648, (128, 128), (1, 128), 0), view_666, out=buf650)
        del view_666
        buf652 = reinterpret_tensor(buf649, (1, 128, 512), (65536, 512, 1), 0); del buf649  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf652, le_31, 65536, grid=grid(65536), stream=stream0)
        del le_31
        buf653 = buf643; del buf643  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf652, (128, 512), (512, 1), 0), permute_1001, out=buf653)
        del permute_1001
        buf658 = reinterpret_tensor(buf633, (1, 128, 128), (16384, 128, 1), 0); del buf633  # reuse
        buf683 = buf628; del buf628  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf648, buf653, primals_263, primals_259, buf658, buf683, 16384, grid=grid(16384), stream=stream0)
        del primals_263
        buf651 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf656 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf657 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf661 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf681 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf682 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_246, layer_input_84, mul_130], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf648, buf653, addmm_246, addmm_241, primals_259, primals_260, buf658, buf651, buf656, buf657, buf661, buf681, buf682, 128, 128, grid=grid(128), stream=stream0)
        del addmm_241
        del addmm_246
        del primals_259
        del primals_260
        buf654 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf652, (512, 128), (1, 512), 0), view_664, out=buf654)
        del view_664
        buf655 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf652, buf655, 512, 128, grid=grid(512), stream=stream0)
        buf659 = buf653; del buf653  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf658, (128, 128), (128, 1), 0), permute_1005, out=buf659)
        del permute_1005
        buf660 = reinterpret_tensor(buf648, (128, 128), (128, 1), 0); del buf648  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf658, (128, 128), (1, 128), 0), view_662, out=buf660)
        del view_662
        # Source Nodes: [], Original ATen: []
        buf662 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf659, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_21, clone_default_22, clone_default_23, None, alias_default_15, getitem_100, getitem_101, getitem_102, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_15
        del clone_default_21
        del clone_default_22
        del clone_default_23
        del getitem_100
        del getitem_101
        del getitem_102
        buf663 = buf662[0]
        buf664 = buf662[1]
        buf665 = buf662[2]
        del buf662
        buf666 = reinterpret_tensor(buf652, (128, 512), (512, 1), 0); del buf652  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf665, (128, 128), (128, 1), 0), permute_1018, out=buf666)
        del permute_1018
        buf667 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf665, (128, 128), (1, 128), 0), view_642, out=buf667)
        buf668 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf665, buf668, 128, 128, grid=grid(128), stream=stream0)
        buf669 = reinterpret_tensor(buf665, (128, 128), (128, 1), 0); del buf665  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf664, (128, 128), (128, 1), 0), permute_1022, out=buf669)
        del permute_1022
        buf670 = buf659; del buf659  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf664, (128, 128), (1, 128), 0), view_646, out=buf670)
        buf671 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf664, buf671, 128, 128, grid=grid(128), stream=stream0)
        buf672 = reinterpret_tensor(buf664, (128, 128), (128, 1), 0); del buf664  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf663, (128, 128), (128, 1), 0), permute_1026, out=buf672)
        del permute_1026
        buf673 = reinterpret_tensor(buf658, (128, 128), (128, 1), 0); del buf658  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf663, (128, 128), (1, 128), 0), view_646, out=buf673)
        del view_646
        buf674 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf663, buf674, 128, 128, grid=grid(128), stream=stream0)
        del buf663
        buf675 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf676 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf669, buf672, addmm_242, buf675, buf676, 128, 128, grid=grid(128), stream=stream0)
        del addmm_242
        buf677 = reinterpret_tensor(buf669, (1, 128, 128), (16384, 128, 1), 0); del buf669  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf677, buf672, primals_261, 16384, grid=grid(16384), stream=stream0)
        del primals_261
        buf678 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf677, (128, 128), (128, 1), 0), permute_1030, out=buf678)
        del permute_1030
        buf679 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf677, (128, 128), (1, 128), 0), view_642, out=buf679)
        buf680 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf677, buf680, 128, 128, grid=grid(128), stream=stream0)
        buf684 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf683, (128, 128), (128, 1), 0), permute_1034, out=buf684)
        del permute_1034
        buf685 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf683, (128, 128), (1, 128), 0), view_642, out=buf685)
        del view_642
        buf686 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf683, buf686, 128, 128, grid=grid(128), stream=stream0)
        buf687 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf689 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_241, mul_121, value_tensor_15], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_24.run(buf611, buf666, buf678, buf684, addmm_240, buf36, primals_241, primals_242, buf687, buf689, 512, 128, grid=grid(512), stream=stream0)
        del addmm_240
        del primals_242
        buf688 = buf611; del buf611  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf688, buf666, buf678, buf684, primals_257, 65536, grid=grid(65536), stream=stream0)
        del primals_257
        buf690 = reinterpret_tensor(buf683, (128, 128), (128, 1), 0); del buf683  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf688, (128, 512), (512, 1), 0), permute_1038, out=buf690)
        del permute_1038
        buf691 = reinterpret_tensor(buf684, (512, 128), (128, 1), 0); del buf684  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf688, (512, 128), (1, 512), 0), view_640, out=buf691)
        del view_640
        buf692 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf688, buf692, 512, 128, grid=grid(512), stream=stream0)
        buf695 = buf677; del buf677  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf690, primals_255, buf695, 16384, grid=grid(16384), stream=stream0)
        del primals_255
        buf696 = buf678; del buf678  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf695, (128, 128), (128, 1), 0), permute_1042, out=buf696)
        del permute_1042
        buf699 = reinterpret_tensor(buf696, (1, 128, 512), (65536, 512, 1), 0); del buf696  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf699, le_32, 65536, grid=grid(65536), stream=stream0)
        del le_32
        buf700 = buf672; del buf672  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf699, (128, 512), (512, 1), 0), permute_1046, out=buf700)
        del permute_1046
        buf693 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf694 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf698 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf703 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf704 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_239, attention_output_78, mul_127], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf690, addmm_239, buf38, primals_253, primals_254, buf695, buf700, buf693, buf694, buf698, buf703, buf704, 128, 128, grid=grid(128), stream=stream0)
        del addmm_239
        del primals_254
        buf697 = buf666; del buf666  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf695, (128, 128), (1, 128), 0), view_638, out=buf697)
        del view_638
        buf701 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf699, (512, 128), (1, 512), 0), view_636, out=buf701)
        del view_636
        buf702 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf699, buf702, 512, 128, grid=grid(512), stream=stream0)
        buf705 = buf695; del buf695  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf705, buf700, primals_253, 16384, grid=grid(16384), stream=stream0)
        del primals_253
        buf706 = reinterpret_tensor(buf699, (128, 512), (512, 1), 0); del buf699  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf705, (128, 128), (128, 1), 0), permute_1050, out=buf706)
        del permute_1050
        buf707 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf705, (128, 128), (1, 128), 0), view_634, out=buf707)
        del view_634
        buf709 = reinterpret_tensor(buf706, (1, 128, 512), (65536, 512, 1), 0); del buf706  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf709, le_33, 65536, grid=grid(65536), stream=stream0)
        del le_33
        buf710 = buf700; del buf700  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf709, (128, 512), (512, 1), 0), permute_1054, out=buf710)
        del permute_1054
        buf715 = reinterpret_tensor(buf690, (1, 128, 128), (16384, 128, 1), 0); del buf690  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf705, buf710, primals_251, buf715, 16384, grid=grid(16384), stream=stream0)
        del primals_251
        buf716 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf715, (128, 128), (128, 1), 0), permute_1058, out=buf716)
        del permute_1058
        buf719 = reinterpret_tensor(buf716, (1, 128, 512), (65536, 512, 1), 0); del buf716  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf719, le_34, 65536, grid=grid(65536), stream=stream0)
        del le_34
        buf720 = reinterpret_tensor(buf38, (128, 128), (128, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf719, (128, 512), (512, 1), 0), permute_1062, out=buf720)
        del permute_1062
        buf708 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf713 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf714 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf718 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf723 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf724 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_235, attention_output_76, mul_125], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf705, buf710, addmm_235, buf37, primals_249, primals_250, buf715, buf720, buf708, buf713, buf714, buf718, buf723, buf724, 128, 128, grid=grid(128), stream=stream0)
        del addmm_235
        del buf37
        del primals_250
        buf711 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf709, (512, 128), (1, 512), 0), view_632, out=buf711)
        del view_632
        buf712 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf709, buf712, 512, 128, grid=grid(512), stream=stream0)
        buf717 = reinterpret_tensor(buf709, (128, 512), (512, 1), 0); del buf709  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf715, (128, 128), (1, 128), 0), view_630, out=buf717)
        del view_630
        buf721 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf719, (512, 128), (1, 512), 0), view_628, out=buf721)
        del view_628
        buf722 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf719, buf722, 512, 128, grid=grid(512), stream=stream0)
        buf725 = buf715; del buf715  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf725, buf720, primals_249, 16384, grid=grid(16384), stream=stream0)
        del primals_249
        buf726 = reinterpret_tensor(buf719, (128, 512), (512, 1), 0); del buf719  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf725, (128, 128), (128, 1), 0), permute_1066, out=buf726)
        del permute_1066
        buf727 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf725, (128, 128), (1, 128), 0), view_626, out=buf727)
        del view_626
        buf729 = reinterpret_tensor(buf726, (1, 128, 512), (65536, 512, 1), 0); del buf726  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf729, le_35, 65536, grid=grid(65536), stream=stream0)
        del le_35
        buf730 = buf720; del buf720  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf729, (128, 512), (512, 1), 0), permute_1070, out=buf730)
        del permute_1070
        buf735 = reinterpret_tensor(buf710, (1, 128, 128), (16384, 128, 1), 0); del buf710  # reuse
        buf760 = buf705; del buf705  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf725, buf730, primals_247, primals_243, buf735, buf760, 16384, grid=grid(16384), stream=stream0)
        del primals_247
        buf728 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf733 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf734 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf738 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf758 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf759 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_231, layer_input_79, mul_122], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf725, buf730, addmm_231, addmm_226, primals_243, primals_244, buf735, buf728, buf733, buf734, buf738, buf758, buf759, 128, 128, grid=grid(128), stream=stream0)
        del addmm_226
        del addmm_231
        del primals_243
        del primals_244
        buf731 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf729, (512, 128), (1, 512), 0), view_624, out=buf731)
        del view_624
        buf732 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf729, buf732, 512, 128, grid=grid(512), stream=stream0)
        buf736 = buf730; del buf730  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf735, (128, 128), (128, 1), 0), permute_1074, out=buf736)
        del permute_1074
        buf737 = reinterpret_tensor(buf725, (128, 128), (128, 1), 0); del buf725  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf735, (128, 128), (1, 128), 0), view_622, out=buf737)
        del view_622
        # Source Nodes: [], Original ATen: []
        buf739 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf736, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_24, clone_default_25, clone_default_26, None, alias_default_17, getitem_107, getitem_108, getitem_109, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_17
        del clone_default_24
        del clone_default_25
        del clone_default_26
        del getitem_107
        del getitem_108
        del getitem_109
        buf740 = buf739[0]
        buf741 = buf739[1]
        buf742 = buf739[2]
        del buf739
        buf743 = reinterpret_tensor(buf729, (128, 512), (512, 1), 0); del buf729  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf742, (128, 128), (128, 1), 0), permute_1087, out=buf743)
        del permute_1087
        buf744 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf742, (128, 128), (1, 128), 0), view_602, out=buf744)
        buf745 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf742, buf745, 128, 128, grid=grid(128), stream=stream0)
        buf746 = reinterpret_tensor(buf742, (128, 128), (128, 1), 0); del buf742  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf741, (128, 128), (128, 1), 0), permute_1091, out=buf746)
        del permute_1091
        buf747 = buf736; del buf736  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf741, (128, 128), (1, 128), 0), view_606, out=buf747)
        buf748 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf741, buf748, 128, 128, grid=grid(128), stream=stream0)
        buf749 = reinterpret_tensor(buf741, (128, 128), (128, 1), 0); del buf741  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf740, (128, 128), (128, 1), 0), permute_1095, out=buf749)
        del permute_1095
        buf750 = reinterpret_tensor(buf735, (128, 128), (128, 1), 0); del buf735  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf740, (128, 128), (1, 128), 0), view_606, out=buf750)
        del view_606
        buf751 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf740, buf751, 128, 128, grid=grid(128), stream=stream0)
        del buf740
        buf752 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf753 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf746, buf749, addmm_227, buf752, buf753, 128, 128, grid=grid(128), stream=stream0)
        del addmm_227
        buf754 = reinterpret_tensor(buf746, (1, 128, 128), (16384, 128, 1), 0); del buf746  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf754, buf749, primals_245, 16384, grid=grid(16384), stream=stream0)
        del primals_245
        buf755 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf754, (128, 128), (128, 1), 0), permute_1099, out=buf755)
        del permute_1099
        buf756 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf754, (128, 128), (1, 128), 0), view_602, out=buf756)
        buf757 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf754, buf757, 128, 128, grid=grid(128), stream=stream0)
        buf761 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf760, (128, 128), (128, 1), 0), permute_1103, out=buf761)
        del permute_1103
        buf762 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf760, (128, 128), (1, 128), 0), view_602, out=buf762)
        del view_602
        buf763 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf760, buf763, 128, 128, grid=grid(128), stream=stream0)
        buf764 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf766 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_22.run(buf688, buf743, buf755, buf761, buf36, buf764, buf766, 512, 128, grid=grid(512), stream=stream0)
        buf765 = buf688; del buf688  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf765, buf743, buf755, buf761, primals_241, 65536, grid=grid(65536), stream=stream0)
        del primals_241
        buf767 = reinterpret_tensor(buf760, (128, 128), (128, 1), 0); del buf760  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf765, (128, 512), (512, 1), 0), permute_1107, out=buf767)
        del permute_1107
        buf768 = reinterpret_tensor(buf761, (512, 128), (128, 1), 0); del buf761  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf765, (512, 128), (1, 512), 0), view_600, out=buf768)
        del view_600
        buf769 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf765, buf769, 512, 128, grid=grid(512), stream=stream0)
        buf772 = buf754; del buf754  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf767, primals_239, buf772, 16384, grid=grid(16384), stream=stream0)
        del primals_239
        buf773 = buf755; del buf755  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf772, (128, 128), (128, 1), 0), permute_1111, out=buf773)
        del permute_1111
        buf776 = reinterpret_tensor(buf773, (1, 128, 512), (65536, 512, 1), 0); del buf773  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf776, le_36, 65536, grid=grid(65536), stream=stream0)
        del le_36
        buf777 = buf749; del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf776, (128, 512), (512, 1), 0), permute_1115, out=buf777)
        del permute_1115
        buf770 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf771 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf775 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf780 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf781 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_224, attention_output_73, mul_119], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf767, addmm_224, buf35, primals_237, primals_238, buf772, buf777, buf770, buf771, buf775, buf780, buf781, 128, 128, grid=grid(128), stream=stream0)
        del addmm_224
        del primals_238
        buf774 = buf743; del buf743  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf772, (128, 128), (1, 128), 0), view_598, out=buf774)
        del view_598
        buf778 = reinterpret_tensor(buf36, (512, 128), (128, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf776, (512, 128), (1, 512), 0), view_596, out=buf778)
        del view_596
        buf779 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf776, buf779, 512, 128, grid=grid(512), stream=stream0)
        buf782 = buf772; del buf772  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf782, buf777, primals_237, 16384, grid=grid(16384), stream=stream0)
        del primals_237
        buf783 = reinterpret_tensor(buf776, (128, 512), (512, 1), 0); del buf776  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf782, (128, 128), (128, 1), 0), permute_1119, out=buf783)
        del permute_1119
        buf784 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf782, (128, 128), (1, 128), 0), view_594, out=buf784)
        del view_594
        buf786 = reinterpret_tensor(buf783, (1, 128, 512), (65536, 512, 1), 0); del buf783  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf786, le_37, 65536, grid=grid(65536), stream=stream0)
        del le_37
        buf787 = buf777; del buf777  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf786, (128, 512), (512, 1), 0), permute_1123, out=buf787)
        del permute_1123
        buf792 = reinterpret_tensor(buf767, (1, 128, 128), (16384, 128, 1), 0); del buf767  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf782, buf787, primals_235, buf792, 16384, grid=grid(16384), stream=stream0)
        del primals_235
        buf793 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf792, (128, 128), (128, 1), 0), permute_1127, out=buf793)
        del permute_1127
        buf796 = reinterpret_tensor(buf793, (1, 128, 512), (65536, 512, 1), 0); del buf793  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf796, le_38, 65536, grid=grid(65536), stream=stream0)
        del le_38
        buf797 = reinterpret_tensor(buf35, (128, 128), (128, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf796, (128, 512), (512, 1), 0), permute_1131, out=buf797)
        del permute_1131
        buf785 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf790 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf791 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf795 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf800 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf801 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_220, attention_output_71, mul_117], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf782, buf787, addmm_220, buf34, primals_233, primals_234, buf792, buf797, buf785, buf790, buf791, buf795, buf800, buf801, 128, 128, grid=grid(128), stream=stream0)
        del addmm_220
        del buf34
        del primals_234
        buf788 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf786, (512, 128), (1, 512), 0), view_592, out=buf788)
        del view_592
        buf789 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf786, buf789, 512, 128, grid=grid(512), stream=stream0)
        buf794 = reinterpret_tensor(buf786, (128, 512), (512, 1), 0); del buf786  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf792, (128, 128), (1, 128), 0), view_590, out=buf794)
        del view_590
        buf798 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf796, (512, 128), (1, 512), 0), view_588, out=buf798)
        del view_588
        buf799 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf796, buf799, 512, 128, grid=grid(512), stream=stream0)
        buf802 = buf792; del buf792  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf802, buf797, primals_233, 16384, grid=grid(16384), stream=stream0)
        del primals_233
        buf803 = reinterpret_tensor(buf796, (128, 512), (512, 1), 0); del buf796  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf802, (128, 128), (128, 1), 0), permute_1135, out=buf803)
        del permute_1135
        buf804 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf802, (128, 128), (1, 128), 0), view_586, out=buf804)
        del view_586
        buf806 = reinterpret_tensor(buf803, (1, 128, 512), (65536, 512, 1), 0); del buf803  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf806, le_39, 65536, grid=grid(65536), stream=stream0)
        del le_39
        buf807 = buf797; del buf797  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf806, (128, 512), (512, 1), 0), permute_1139, out=buf807)
        del permute_1139
        buf812 = reinterpret_tensor(buf787, (1, 128, 128), (16384, 128, 1), 0); del buf787  # reuse
        buf837 = buf782; del buf782  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf802, buf807, primals_231, primals_227, buf812, buf837, 16384, grid=grid(16384), stream=stream0)
        del primals_231
        buf805 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf810 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf811 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf815 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf835 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf836 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_216, layer_input_74, mul_114], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf802, buf807, addmm_216, addmm_211, primals_227, primals_228, buf812, buf805, buf810, buf811, buf815, buf835, buf836, 128, 128, grid=grid(128), stream=stream0)
        del addmm_211
        del addmm_216
        del primals_227
        del primals_228
        buf808 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf806, (512, 128), (1, 512), 0), view_584, out=buf808)
        del view_584
        buf809 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf806, buf809, 512, 128, grid=grid(512), stream=stream0)
        buf813 = buf807; del buf807  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf812, (128, 128), (128, 1), 0), permute_1143, out=buf813)
        del permute_1143
        buf814 = reinterpret_tensor(buf802, (128, 128), (128, 1), 0); del buf802  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf812, (128, 128), (1, 128), 0), view_582, out=buf814)
        del view_582
        # Source Nodes: [], Original ATen: []
        buf816 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf813, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_27, clone_default_28, clone_default_29, None, alias_default_19, getitem_114, getitem_115, getitem_116, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_19
        del clone_default_27
        del clone_default_28
        del clone_default_29
        del getitem_114
        del getitem_115
        del getitem_116
        buf817 = buf816[0]
        buf818 = buf816[1]
        buf819 = buf816[2]
        del buf816
        buf820 = reinterpret_tensor(buf806, (128, 512), (512, 1), 0); del buf806  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf819, (128, 128), (128, 1), 0), permute_1156, out=buf820)
        del permute_1156
        buf821 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf819, (128, 128), (1, 128), 0), view_562, out=buf821)
        buf822 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf819, buf822, 128, 128, grid=grid(128), stream=stream0)
        buf823 = reinterpret_tensor(buf819, (128, 128), (128, 1), 0); del buf819  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf818, (128, 128), (128, 1), 0), permute_1160, out=buf823)
        del permute_1160
        buf824 = buf813; del buf813  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf818, (128, 128), (1, 128), 0), view_566, out=buf824)
        buf825 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf818, buf825, 128, 128, grid=grid(128), stream=stream0)
        buf826 = reinterpret_tensor(buf818, (128, 128), (128, 1), 0); del buf818  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf817, (128, 128), (128, 1), 0), permute_1164, out=buf826)
        del permute_1164
        buf827 = reinterpret_tensor(buf812, (128, 128), (128, 1), 0); del buf812  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf817, (128, 128), (1, 128), 0), view_566, out=buf827)
        del view_566
        buf828 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf817, buf828, 128, 128, grid=grid(128), stream=stream0)
        del buf817
        buf829 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf830 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf823, buf826, addmm_212, buf829, buf830, 128, 128, grid=grid(128), stream=stream0)
        del addmm_212
        buf831 = reinterpret_tensor(buf823, (1, 128, 128), (16384, 128, 1), 0); del buf823  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf831, buf826, primals_229, 16384, grid=grid(16384), stream=stream0)
        del primals_229
        buf832 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf831, (128, 128), (128, 1), 0), permute_1168, out=buf832)
        del permute_1168
        buf833 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf831, (128, 128), (1, 128), 0), view_562, out=buf833)
        buf834 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf831, buf834, 128, 128, grid=grid(128), stream=stream0)
        buf838 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf837, (128, 128), (128, 1), 0), permute_1172, out=buf838)
        del permute_1172
        buf839 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf837, (128, 128), (1, 128), 0), view_562, out=buf839)
        del view_562
        buf840 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf837, buf840, 128, 128, grid=grid(128), stream=stream0)
        buf841 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf843 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_211, mul_105, value_tensor_13], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_24.run(buf765, buf820, buf832, buf838, addmm_210, buf31, primals_209, primals_210, buf841, buf843, 512, 128, grid=grid(512), stream=stream0)
        del addmm_210
        del primals_210
        buf842 = buf765; del buf765  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf842, buf820, buf832, buf838, primals_225, 65536, grid=grid(65536), stream=stream0)
        del primals_225
        buf844 = reinterpret_tensor(buf837, (128, 128), (128, 1), 0); del buf837  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf842, (128, 512), (512, 1), 0), permute_1176, out=buf844)
        del permute_1176
        buf845 = reinterpret_tensor(buf838, (512, 128), (128, 1), 0); del buf838  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf842, (512, 128), (1, 512), 0), view_560, out=buf845)
        del view_560
        buf846 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf842, buf846, 512, 128, grid=grid(512), stream=stream0)
        buf849 = buf831; del buf831  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf844, primals_223, buf849, 16384, grid=grid(16384), stream=stream0)
        del primals_223
        buf850 = buf832; del buf832  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf849, (128, 128), (128, 1), 0), permute_1180, out=buf850)
        del permute_1180
        buf853 = reinterpret_tensor(buf850, (1, 128, 512), (65536, 512, 1), 0); del buf850  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf853, le_40, 65536, grid=grid(65536), stream=stream0)
        del le_40
        buf854 = buf826; del buf826  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf853, (128, 512), (512, 1), 0), permute_1184, out=buf854)
        del permute_1184
        buf847 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf848 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf852 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf857 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf858 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_209, attention_output_68, mul_111], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf844, addmm_209, buf33, primals_221, primals_222, buf849, buf854, buf847, buf848, buf852, buf857, buf858, 128, 128, grid=grid(128), stream=stream0)
        del addmm_209
        del primals_222
        buf851 = buf820; del buf820  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf849, (128, 128), (1, 128), 0), view_558, out=buf851)
        del view_558
        buf855 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf853, (512, 128), (1, 512), 0), view_556, out=buf855)
        del view_556
        buf856 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf853, buf856, 512, 128, grid=grid(512), stream=stream0)
        buf859 = buf849; del buf849  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf859, buf854, primals_221, 16384, grid=grid(16384), stream=stream0)
        del primals_221
        buf860 = reinterpret_tensor(buf853, (128, 512), (512, 1), 0); del buf853  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf859, (128, 128), (128, 1), 0), permute_1188, out=buf860)
        del permute_1188
        buf861 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf859, (128, 128), (1, 128), 0), view_554, out=buf861)
        del view_554
        buf863 = reinterpret_tensor(buf860, (1, 128, 512), (65536, 512, 1), 0); del buf860  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf863, le_41, 65536, grid=grid(65536), stream=stream0)
        del le_41
        buf864 = buf854; del buf854  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf863, (128, 512), (512, 1), 0), permute_1192, out=buf864)
        del permute_1192
        buf869 = reinterpret_tensor(buf844, (1, 128, 128), (16384, 128, 1), 0); del buf844  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf859, buf864, primals_219, buf869, 16384, grid=grid(16384), stream=stream0)
        del primals_219
        buf870 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf869, (128, 128), (128, 1), 0), permute_1196, out=buf870)
        del permute_1196
        buf873 = reinterpret_tensor(buf870, (1, 128, 512), (65536, 512, 1), 0); del buf870  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf873, le_42, 65536, grid=grid(65536), stream=stream0)
        del le_42
        buf874 = reinterpret_tensor(buf33, (128, 128), (128, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf873, (128, 512), (512, 1), 0), permute_1200, out=buf874)
        del permute_1200
        buf862 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf867 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf868 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf872 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf877 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf878 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_205, attention_output_66, mul_109], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf859, buf864, addmm_205, buf32, primals_217, primals_218, buf869, buf874, buf862, buf867, buf868, buf872, buf877, buf878, 128, 128, grid=grid(128), stream=stream0)
        del addmm_205
        del buf32
        del primals_218
        buf865 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf863, (512, 128), (1, 512), 0), view_552, out=buf865)
        del view_552
        buf866 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf863, buf866, 512, 128, grid=grid(512), stream=stream0)
        buf871 = reinterpret_tensor(buf863, (128, 512), (512, 1), 0); del buf863  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf869, (128, 128), (1, 128), 0), view_550, out=buf871)
        del view_550
        buf875 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf873, (512, 128), (1, 512), 0), view_548, out=buf875)
        del view_548
        buf876 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf873, buf876, 512, 128, grid=grid(512), stream=stream0)
        buf879 = buf869; del buf869  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf879, buf874, primals_217, 16384, grid=grid(16384), stream=stream0)
        del primals_217
        buf880 = reinterpret_tensor(buf873, (128, 512), (512, 1), 0); del buf873  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf879, (128, 128), (128, 1), 0), permute_1204, out=buf880)
        del permute_1204
        buf881 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf879, (128, 128), (1, 128), 0), view_546, out=buf881)
        del view_546
        buf883 = reinterpret_tensor(buf880, (1, 128, 512), (65536, 512, 1), 0); del buf880  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf883, le_43, 65536, grid=grid(65536), stream=stream0)
        del le_43
        buf884 = buf874; del buf874  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf883, (128, 512), (512, 1), 0), permute_1208, out=buf884)
        del permute_1208
        buf889 = reinterpret_tensor(buf864, (1, 128, 128), (16384, 128, 1), 0); del buf864  # reuse
        buf914 = buf859; del buf859  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf879, buf884, primals_215, primals_211, buf889, buf914, 16384, grid=grid(16384), stream=stream0)
        del primals_215
        buf882 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf887 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf888 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf892 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf912 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf913 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_201, layer_input_69, mul_106], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf879, buf884, addmm_201, addmm_196, primals_211, primals_212, buf889, buf882, buf887, buf888, buf892, buf912, buf913, 128, 128, grid=grid(128), stream=stream0)
        del addmm_196
        del addmm_201
        del primals_211
        del primals_212
        buf885 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf883, (512, 128), (1, 512), 0), view_544, out=buf885)
        del view_544
        buf886 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf883, buf886, 512, 128, grid=grid(512), stream=stream0)
        buf890 = buf884; del buf884  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf889, (128, 128), (128, 1), 0), permute_1212, out=buf890)
        del permute_1212
        buf891 = reinterpret_tensor(buf879, (128, 128), (128, 1), 0); del buf879  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf889, (128, 128), (1, 128), 0), view_542, out=buf891)
        del view_542
        # Source Nodes: [], Original ATen: []
        buf893 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf890, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_30, clone_default_31, clone_default_32, None, alias_default_21, getitem_121, getitem_122, getitem_123, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_21
        del clone_default_30
        del clone_default_31
        del clone_default_32
        del getitem_121
        del getitem_122
        del getitem_123
        buf894 = buf893[0]
        buf895 = buf893[1]
        buf896 = buf893[2]
        del buf893
        buf897 = reinterpret_tensor(buf883, (128, 512), (512, 1), 0); del buf883  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf896, (128, 128), (128, 1), 0), permute_1225, out=buf897)
        del permute_1225
        buf898 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf896, (128, 128), (1, 128), 0), view_522, out=buf898)
        buf899 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf896, buf899, 128, 128, grid=grid(128), stream=stream0)
        buf900 = reinterpret_tensor(buf896, (128, 128), (128, 1), 0); del buf896  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf895, (128, 128), (128, 1), 0), permute_1229, out=buf900)
        del permute_1229
        buf901 = buf890; del buf890  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf895, (128, 128), (1, 128), 0), view_526, out=buf901)
        buf902 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf895, buf902, 128, 128, grid=grid(128), stream=stream0)
        buf903 = reinterpret_tensor(buf895, (128, 128), (128, 1), 0); del buf895  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf894, (128, 128), (128, 1), 0), permute_1233, out=buf903)
        del permute_1233
        buf904 = reinterpret_tensor(buf889, (128, 128), (128, 1), 0); del buf889  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf894, (128, 128), (1, 128), 0), view_526, out=buf904)
        del view_526
        buf905 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf894, buf905, 128, 128, grid=grid(128), stream=stream0)
        del buf894
        buf906 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf907 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf900, buf903, addmm_197, buf906, buf907, 128, 128, grid=grid(128), stream=stream0)
        del addmm_197
        buf908 = reinterpret_tensor(buf900, (1, 128, 128), (16384, 128, 1), 0); del buf900  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf908, buf903, primals_213, 16384, grid=grid(16384), stream=stream0)
        del primals_213
        buf909 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf908, (128, 128), (128, 1), 0), permute_1237, out=buf909)
        del permute_1237
        buf910 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf908, (128, 128), (1, 128), 0), view_522, out=buf910)
        buf911 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf908, buf911, 128, 128, grid=grid(128), stream=stream0)
        buf915 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf914, (128, 128), (128, 1), 0), permute_1241, out=buf915)
        del permute_1241
        buf916 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf914, (128, 128), (1, 128), 0), view_522, out=buf916)
        del view_522
        buf917 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf914, buf917, 128, 128, grid=grid(128), stream=stream0)
        buf918 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf920 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_22.run(buf842, buf897, buf909, buf915, buf31, buf918, buf920, 512, 128, grid=grid(512), stream=stream0)
        buf919 = buf842; del buf842  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf919, buf897, buf909, buf915, primals_209, 65536, grid=grid(65536), stream=stream0)
        del primals_209
        buf921 = reinterpret_tensor(buf914, (128, 128), (128, 1), 0); del buf914  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf919, (128, 512), (512, 1), 0), permute_1245, out=buf921)
        del permute_1245
        buf922 = reinterpret_tensor(buf915, (512, 128), (128, 1), 0); del buf915  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf919, (512, 128), (1, 512), 0), view_520, out=buf922)
        del view_520
        buf923 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf919, buf923, 512, 128, grid=grid(512), stream=stream0)
        buf926 = buf908; del buf908  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf921, primals_207, buf926, 16384, grid=grid(16384), stream=stream0)
        del primals_207
        buf927 = buf909; del buf909  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf926, (128, 128), (128, 1), 0), permute_1249, out=buf927)
        del permute_1249
        buf930 = reinterpret_tensor(buf927, (1, 128, 512), (65536, 512, 1), 0); del buf927  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf930, le_44, 65536, grid=grid(65536), stream=stream0)
        del le_44
        buf931 = buf903; del buf903  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf930, (128, 512), (512, 1), 0), permute_1253, out=buf931)
        del permute_1253
        buf924 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf925 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf929 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf934 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf935 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_194, attention_output_63, mul_103], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf921, addmm_194, buf30, primals_205, primals_206, buf926, buf931, buf924, buf925, buf929, buf934, buf935, 128, 128, grid=grid(128), stream=stream0)
        del addmm_194
        del primals_206
        buf928 = buf897; del buf897  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf926, (128, 128), (1, 128), 0), view_518, out=buf928)
        del view_518
        buf932 = reinterpret_tensor(buf31, (512, 128), (128, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf930, (512, 128), (1, 512), 0), view_516, out=buf932)
        del view_516
        buf933 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf930, buf933, 512, 128, grid=grid(512), stream=stream0)
        buf936 = buf926; del buf926  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf936, buf931, primals_205, 16384, grid=grid(16384), stream=stream0)
        del primals_205
        buf937 = reinterpret_tensor(buf930, (128, 512), (512, 1), 0); del buf930  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf936, (128, 128), (128, 1), 0), permute_1257, out=buf937)
        del permute_1257
        buf938 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf936, (128, 128), (1, 128), 0), view_514, out=buf938)
        del view_514
        buf940 = reinterpret_tensor(buf937, (1, 128, 512), (65536, 512, 1), 0); del buf937  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf940, le_45, 65536, grid=grid(65536), stream=stream0)
        del le_45
        buf941 = buf931; del buf931  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf940, (128, 512), (512, 1), 0), permute_1261, out=buf941)
        del permute_1261
        buf946 = reinterpret_tensor(buf921, (1, 128, 128), (16384, 128, 1), 0); del buf921  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf936, buf941, primals_203, buf946, 16384, grid=grid(16384), stream=stream0)
        del primals_203
        buf947 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf946, (128, 128), (128, 1), 0), permute_1265, out=buf947)
        del permute_1265
        buf950 = reinterpret_tensor(buf947, (1, 128, 512), (65536, 512, 1), 0); del buf947  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf950, le_46, 65536, grid=grid(65536), stream=stream0)
        del le_46
        buf951 = reinterpret_tensor(buf30, (128, 128), (128, 1), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf950, (128, 512), (512, 1), 0), permute_1269, out=buf951)
        del permute_1269
        buf939 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf944 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf945 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf949 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf954 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf955 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_190, attention_output_61, mul_101], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf936, buf941, addmm_190, buf29, primals_201, primals_202, buf946, buf951, buf939, buf944, buf945, buf949, buf954, buf955, 128, 128, grid=grid(128), stream=stream0)
        del addmm_190
        del buf29
        del primals_202
        buf942 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf940, (512, 128), (1, 512), 0), view_512, out=buf942)
        del view_512
        buf943 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf940, buf943, 512, 128, grid=grid(512), stream=stream0)
        buf948 = reinterpret_tensor(buf940, (128, 512), (512, 1), 0); del buf940  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf946, (128, 128), (1, 128), 0), view_510, out=buf948)
        del view_510
        buf952 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf950, (512, 128), (1, 512), 0), view_508, out=buf952)
        del view_508
        buf953 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf950, buf953, 512, 128, grid=grid(512), stream=stream0)
        buf956 = buf946; del buf946  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf956, buf951, primals_201, 16384, grid=grid(16384), stream=stream0)
        del primals_201
        buf957 = reinterpret_tensor(buf950, (128, 512), (512, 1), 0); del buf950  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf956, (128, 128), (128, 1), 0), permute_1273, out=buf957)
        del permute_1273
        buf958 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf956, (128, 128), (1, 128), 0), view_506, out=buf958)
        del view_506
        buf960 = reinterpret_tensor(buf957, (1, 128, 512), (65536, 512, 1), 0); del buf957  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf960, le_47, 65536, grid=grid(65536), stream=stream0)
        del le_47
        buf961 = buf951; del buf951  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf960, (128, 512), (512, 1), 0), permute_1277, out=buf961)
        del permute_1277
        buf966 = reinterpret_tensor(buf941, (1, 128, 128), (16384, 128, 1), 0); del buf941  # reuse
        buf991 = buf936; del buf936  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf956, buf961, primals_199, primals_195, buf966, buf991, 16384, grid=grid(16384), stream=stream0)
        del primals_199
        buf959 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf964 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf965 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf969 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf989 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf990 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_186, layer_input_64, mul_98], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf956, buf961, addmm_186, addmm_181, primals_195, primals_196, buf966, buf959, buf964, buf965, buf969, buf989, buf990, 128, 128, grid=grid(128), stream=stream0)
        del addmm_181
        del addmm_186
        del primals_195
        del primals_196
        buf962 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf960, (512, 128), (1, 512), 0), view_504, out=buf962)
        del view_504
        buf963 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf960, buf963, 512, 128, grid=grid(512), stream=stream0)
        buf967 = buf961; del buf961  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf966, (128, 128), (128, 1), 0), permute_1281, out=buf967)
        del permute_1281
        buf968 = reinterpret_tensor(buf956, (128, 128), (128, 1), 0); del buf956  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf966, (128, 128), (1, 128), 0), view_502, out=buf968)
        del view_502
        # Source Nodes: [], Original ATen: []
        buf970 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf967, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_33, clone_default_34, clone_default_35, None, alias_default_23, getitem_128, getitem_129, getitem_130, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_23
        del clone_default_33
        del clone_default_34
        del clone_default_35
        del getitem_128
        del getitem_129
        del getitem_130
        buf971 = buf970[0]
        buf972 = buf970[1]
        buf973 = buf970[2]
        del buf970
        buf974 = reinterpret_tensor(buf960, (128, 512), (512, 1), 0); del buf960  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf973, (128, 128), (128, 1), 0), permute_1294, out=buf974)
        del permute_1294
        buf975 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf973, (128, 128), (1, 128), 0), view_482, out=buf975)
        buf976 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf973, buf976, 128, 128, grid=grid(128), stream=stream0)
        buf977 = reinterpret_tensor(buf973, (128, 128), (128, 1), 0); del buf973  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf972, (128, 128), (128, 1), 0), permute_1298, out=buf977)
        del permute_1298
        buf978 = buf967; del buf967  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf972, (128, 128), (1, 128), 0), view_486, out=buf978)
        buf979 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf972, buf979, 128, 128, grid=grid(128), stream=stream0)
        buf980 = reinterpret_tensor(buf972, (128, 128), (128, 1), 0); del buf972  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf971, (128, 128), (128, 1), 0), permute_1302, out=buf980)
        del permute_1302
        buf981 = reinterpret_tensor(buf966, (128, 128), (128, 1), 0); del buf966  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf971, (128, 128), (1, 128), 0), view_486, out=buf981)
        del view_486
        buf982 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf971, buf982, 128, 128, grid=grid(128), stream=stream0)
        del buf971
        buf983 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf984 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf977, buf980, addmm_182, buf983, buf984, 128, 128, grid=grid(128), stream=stream0)
        del addmm_182
        buf985 = reinterpret_tensor(buf977, (1, 128, 128), (16384, 128, 1), 0); del buf977  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf985, buf980, primals_197, 16384, grid=grid(16384), stream=stream0)
        del primals_197
        buf986 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf985, (128, 128), (128, 1), 0), permute_1306, out=buf986)
        del permute_1306
        buf987 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf985, (128, 128), (1, 128), 0), view_482, out=buf987)
        buf988 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf985, buf988, 128, 128, grid=grid(128), stream=stream0)
        buf992 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf991, (128, 128), (128, 1), 0), permute_1310, out=buf992)
        del permute_1310
        buf993 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf991, (128, 128), (1, 128), 0), view_482, out=buf993)
        del view_482
        buf994 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf991, buf994, 128, 128, grid=grid(128), stream=stream0)
        buf995 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf997 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_181, mul_89, value_tensor_11], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_24.run(buf919, buf974, buf986, buf992, addmm_180, buf26, primals_177, primals_178, buf995, buf997, 512, 128, grid=grid(512), stream=stream0)
        del addmm_180
        del primals_178
        buf996 = buf919; del buf919  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf996, buf974, buf986, buf992, primals_193, 65536, grid=grid(65536), stream=stream0)
        del primals_193
        buf998 = reinterpret_tensor(buf991, (128, 128), (128, 1), 0); del buf991  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf996, (128, 512), (512, 1), 0), permute_1314, out=buf998)
        del permute_1314
        buf999 = reinterpret_tensor(buf992, (512, 128), (128, 1), 0); del buf992  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf996, (512, 128), (1, 512), 0), view_480, out=buf999)
        del view_480
        buf1000 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf996, buf1000, 512, 128, grid=grid(512), stream=stream0)
        buf1003 = buf985; del buf985  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf998, primals_191, buf1003, 16384, grid=grid(16384), stream=stream0)
        del primals_191
        buf1004 = buf986; del buf986  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1003, (128, 128), (128, 1), 0), permute_1318, out=buf1004)
        del permute_1318
        buf1007 = reinterpret_tensor(buf1004, (1, 128, 512), (65536, 512, 1), 0); del buf1004  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1007, le_48, 65536, grid=grid(65536), stream=stream0)
        del le_48
        buf1008 = buf980; del buf980  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1007, (128, 512), (512, 1), 0), permute_1322, out=buf1008)
        del permute_1322
        buf1001 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1002 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1006 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1011 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1012 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_179, attention_output_58, mul_95], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf998, addmm_179, buf28, primals_189, primals_190, buf1003, buf1008, buf1001, buf1002, buf1006, buf1011, buf1012, 128, 128, grid=grid(128), stream=stream0)
        del addmm_179
        del primals_190
        buf1005 = buf974; del buf974  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1003, (128, 128), (1, 128), 0), view_478, out=buf1005)
        del view_478
        buf1009 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1007, (512, 128), (1, 512), 0), view_476, out=buf1009)
        del view_476
        buf1010 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1007, buf1010, 512, 128, grid=grid(512), stream=stream0)
        buf1013 = buf1003; del buf1003  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1013, buf1008, primals_189, 16384, grid=grid(16384), stream=stream0)
        del primals_189
        buf1014 = reinterpret_tensor(buf1007, (128, 512), (512, 1), 0); del buf1007  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1013, (128, 128), (128, 1), 0), permute_1326, out=buf1014)
        del permute_1326
        buf1015 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1013, (128, 128), (1, 128), 0), view_474, out=buf1015)
        del view_474
        buf1017 = reinterpret_tensor(buf1014, (1, 128, 512), (65536, 512, 1), 0); del buf1014  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1017, le_49, 65536, grid=grid(65536), stream=stream0)
        del le_49
        buf1018 = buf1008; del buf1008  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1017, (128, 512), (512, 1), 0), permute_1330, out=buf1018)
        del permute_1330
        buf1023 = reinterpret_tensor(buf998, (1, 128, 128), (16384, 128, 1), 0); del buf998  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf1013, buf1018, primals_187, buf1023, 16384, grid=grid(16384), stream=stream0)
        del primals_187
        buf1024 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1023, (128, 128), (128, 1), 0), permute_1334, out=buf1024)
        del permute_1334
        buf1027 = reinterpret_tensor(buf1024, (1, 128, 512), (65536, 512, 1), 0); del buf1024  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1027, le_50, 65536, grid=grid(65536), stream=stream0)
        del le_50
        buf1028 = reinterpret_tensor(buf28, (128, 128), (128, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1027, (128, 512), (512, 1), 0), permute_1338, out=buf1028)
        del permute_1338
        buf1016 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1021 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1022 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1026 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1031 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1032 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_175, attention_output_56, mul_93], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf1013, buf1018, addmm_175, buf27, primals_185, primals_186, buf1023, buf1028, buf1016, buf1021, buf1022, buf1026, buf1031, buf1032, 128, 128, grid=grid(128), stream=stream0)
        del addmm_175
        del buf1013
        del primals_186
        buf1019 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1017, (512, 128), (1, 512), 0), view_472, out=buf1019)
        del view_472
        buf1020 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1017, buf1020, 512, 128, grid=grid(512), stream=stream0)
        buf1025 = reinterpret_tensor(buf1017, (128, 512), (512, 1), 0); del buf1017  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1023, (128, 128), (1, 128), 0), view_470, out=buf1025)
        del view_470
        buf1029 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1027, (512, 128), (1, 512), 0), view_468, out=buf1029)
        del view_468
        buf1030 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1027, buf1030, 512, 128, grid=grid(512), stream=stream0)
        buf1033 = buf1023; del buf1023  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1033, buf1028, primals_185, 16384, grid=grid(16384), stream=stream0)
        del primals_185
        buf1034 = reinterpret_tensor(buf1027, (128, 512), (512, 1), 0); del buf1027  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1033, (128, 128), (128, 1), 0), permute_1342, out=buf1034)
        del permute_1342
        buf1035 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1033, (128, 128), (1, 128), 0), view_466, out=buf1035)
        del view_466
        buf1037 = reinterpret_tensor(buf1034, (1, 128, 512), (65536, 512, 1), 0); del buf1034  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1037, le_51, 65536, grid=grid(65536), stream=stream0)
        del le_51
        buf1038 = buf1028; del buf1028  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1037, (128, 512), (512, 1), 0), permute_1346, out=buf1038)
        del permute_1346
        buf1043 = buf27; del buf27  # reuse
        buf1068 = reinterpret_tensor(buf1018, (1, 128, 128), (16384, 128, 1), 0); del buf1018  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf1033, buf1038, primals_183, primals_179, buf1043, buf1068, 16384, grid=grid(16384), stream=stream0)
        del primals_183
        buf1036 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1041 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1042 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1046 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1066 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1067 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_171, layer_input_59, mul_90], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf1033, buf1038, addmm_171, addmm_166, primals_179, primals_180, buf1043, buf1036, buf1041, buf1042, buf1046, buf1066, buf1067, 128, 128, grid=grid(128), stream=stream0)
        del addmm_166
        del addmm_171
        del primals_179
        del primals_180
        buf1039 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1037, (512, 128), (1, 512), 0), view_464, out=buf1039)
        del view_464
        buf1040 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1037, buf1040, 512, 128, grid=grid(512), stream=stream0)
        buf1044 = buf1038; del buf1038  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1043, (128, 128), (128, 1), 0), permute_1350, out=buf1044)
        del permute_1350
        buf1045 = reinterpret_tensor(buf1033, (128, 128), (128, 1), 0); del buf1033  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1043, (128, 128), (1, 128), 0), view_462, out=buf1045)
        del view_462
        # Source Nodes: [], Original ATen: []
        buf1047 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1044, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_36, clone_default_37, clone_default_38, None, alias_default_25, getitem_135, getitem_136, getitem_137, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_25
        del clone_default_36
        del clone_default_37
        del clone_default_38
        del getitem_135
        del getitem_136
        del getitem_137
        buf1048 = buf1047[0]
        buf1049 = buf1047[1]
        buf1050 = buf1047[2]
        del buf1047
        buf1051 = reinterpret_tensor(buf1037, (128, 512), (512, 1), 0); del buf1037  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1050, (128, 128), (128, 1), 0), permute_1363, out=buf1051)
        del permute_1363
        buf1052 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1050, (128, 128), (1, 128), 0), view_442, out=buf1052)
        buf1053 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1050, buf1053, 128, 128, grid=grid(128), stream=stream0)
        buf1054 = reinterpret_tensor(buf1050, (128, 128), (128, 1), 0); del buf1050  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1049, (128, 128), (128, 1), 0), permute_1367, out=buf1054)
        del permute_1367
        buf1055 = buf1044; del buf1044  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1049, (128, 128), (1, 128), 0), view_446, out=buf1055)
        buf1056 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1049, buf1056, 128, 128, grid=grid(128), stream=stream0)
        buf1057 = reinterpret_tensor(buf1049, (128, 128), (128, 1), 0); del buf1049  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1048, (128, 128), (128, 1), 0), permute_1371, out=buf1057)
        del permute_1371
        buf1058 = reinterpret_tensor(buf1043, (128, 128), (128, 1), 0); del buf1043  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1048, (128, 128), (1, 128), 0), view_446, out=buf1058)
        del view_446
        buf1059 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1048, buf1059, 128, 128, grid=grid(128), stream=stream0)
        del buf1048
        buf1060 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1061 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf1054, buf1057, addmm_167, buf1060, buf1061, 128, 128, grid=grid(128), stream=stream0)
        del addmm_167
        buf1062 = reinterpret_tensor(buf1054, (1, 128, 128), (16384, 128, 1), 0); del buf1054  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1062, buf1057, primals_181, 16384, grid=grid(16384), stream=stream0)
        del primals_181
        buf1063 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1062, (128, 128), (128, 1), 0), permute_1375, out=buf1063)
        del permute_1375
        buf1064 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1062, (128, 128), (1, 128), 0), view_442, out=buf1064)
        buf1065 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1062, buf1065, 128, 128, grid=grid(128), stream=stream0)
        buf1069 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1068, (128, 128), (128, 1), 0), permute_1379, out=buf1069)
        del permute_1379
        buf1070 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1068, (128, 128), (1, 128), 0), view_442, out=buf1070)
        del view_442
        buf1071 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1068, buf1071, 128, 128, grid=grid(128), stream=stream0)
        buf1072 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1074 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_22.run(buf996, buf1051, buf1063, buf1069, buf26, buf1072, buf1074, 512, 128, grid=grid(512), stream=stream0)
        buf1073 = reinterpret_tensor(buf1051, (1, 128, 512), (65536, 512, 1), 0); del buf1051  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_23.run(buf1073, buf996, buf1063, buf1069, primals_177, 65536, grid=grid(65536), stream=stream0)
        del primals_177
        buf1075 = reinterpret_tensor(buf1068, (128, 128), (128, 1), 0); del buf1068  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1073, (128, 512), (512, 1), 0), permute_1383, out=buf1075)
        del permute_1383
        buf1076 = reinterpret_tensor(buf996, (512, 128), (128, 1), 0); del buf996  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1073, (512, 128), (1, 512), 0), view_440, out=buf1076)
        del view_440
        buf1077 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf1073, buf1077, 512, 128, grid=grid(512), stream=stream0)
        buf1080 = buf1062; del buf1062  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf1075, primals_175, buf1080, 16384, grid=grid(16384), stream=stream0)
        del primals_175
        buf1081 = buf1069; del buf1069  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1080, (128, 128), (128, 1), 0), permute_1387, out=buf1081)
        del permute_1387
        buf1084 = reinterpret_tensor(buf1081, (1, 128, 512), (65536, 512, 1), 0); del buf1081  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1084, le_52, 65536, grid=grid(65536), stream=stream0)
        del le_52
        buf1085 = buf1057; del buf1057  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1084, (128, 512), (512, 1), 0), permute_1391, out=buf1085)
        del permute_1391
        buf1078 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1079 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1083 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1088 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1089 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_164, attention_output_53, mul_87], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf1075, addmm_164, buf25, primals_173, primals_174, buf1080, buf1085, buf1078, buf1079, buf1083, buf1088, buf1089, 128, 128, grid=grid(128), stream=stream0)
        del addmm_164
        del primals_174
        buf1082 = buf1063; del buf1063  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1080, (128, 128), (1, 128), 0), view_438, out=buf1082)
        del view_438
        buf1086 = reinterpret_tensor(buf26, (512, 128), (128, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1084, (512, 128), (1, 512), 0), view_436, out=buf1086)
        del view_436
        buf1087 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1084, buf1087, 512, 128, grid=grid(512), stream=stream0)
        buf1090 = buf1080; del buf1080  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1090, buf1085, primals_173, 16384, grid=grid(16384), stream=stream0)
        del primals_173
        buf1091 = reinterpret_tensor(buf1084, (128, 512), (512, 1), 0); del buf1084  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1090, (128, 128), (128, 1), 0), permute_1395, out=buf1091)
        del permute_1395
        buf1092 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1090, (128, 128), (1, 128), 0), view_434, out=buf1092)
        del view_434
        buf1094 = reinterpret_tensor(buf1091, (1, 128, 512), (65536, 512, 1), 0); del buf1091  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1094, le_53, 65536, grid=grid(65536), stream=stream0)
        del le_53
        buf1095 = buf1085; del buf1085  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1094, (128, 512), (512, 1), 0), permute_1399, out=buf1095)
        del permute_1399
        buf1100 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf1090, buf1095, primals_171, buf1100, 16384, grid=grid(16384), stream=stream0)
        del primals_171
        buf1101 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1100, (128, 128), (128, 1), 0), permute_1403, out=buf1101)
        del permute_1403
        buf1104 = reinterpret_tensor(buf1101, (1, 128, 512), (65536, 512, 1), 0); del buf1101  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1104, le_54, 65536, grid=grid(65536), stream=stream0)
        del le_54
        buf1105 = buf1075; del buf1075  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1104, (128, 512), (512, 1), 0), permute_1407, out=buf1105)
        del permute_1407
        buf1093 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1098 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1099 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1103 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1108 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1109 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_160, attention_output_51, mul_85], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf1090, buf1095, addmm_160, buf24, primals_169, primals_170, buf1100, buf1105, buf1093, buf1098, buf1099, buf1103, buf1108, buf1109, 128, 128, grid=grid(128), stream=stream0)
        del addmm_160
        del buf1090
        del primals_170
        buf1096 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1094, (512, 128), (1, 512), 0), view_432, out=buf1096)
        del view_432
        buf1097 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1094, buf1097, 512, 128, grid=grid(512), stream=stream0)
        buf1102 = reinterpret_tensor(buf1094, (128, 512), (512, 1), 0); del buf1094  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1100, (128, 128), (1, 128), 0), view_430, out=buf1102)
        del view_430
        buf1106 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1104, (512, 128), (1, 512), 0), view_428, out=buf1106)
        del view_428
        buf1107 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1104, buf1107, 512, 128, grid=grid(512), stream=stream0)
        buf1110 = buf1100; del buf1100  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1110, buf1105, primals_169, 16384, grid=grid(16384), stream=stream0)
        del primals_169
        buf1111 = reinterpret_tensor(buf1104, (128, 512), (512, 1), 0); del buf1104  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1110, (128, 128), (128, 1), 0), permute_1411, out=buf1111)
        del permute_1411
        buf1112 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1110, (128, 128), (1, 128), 0), view_426, out=buf1112)
        del view_426
        buf1114 = reinterpret_tensor(buf1111, (1, 128, 512), (65536, 512, 1), 0); del buf1111  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1114, le_55, 65536, grid=grid(65536), stream=stream0)
        del le_55
        buf1115 = buf1105; del buf1105  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1114, (128, 512), (512, 1), 0), permute_1415, out=buf1115)
        del permute_1415
        buf1120 = buf24; del buf24  # reuse
        buf1145 = reinterpret_tensor(buf1095, (1, 128, 128), (16384, 128, 1), 0); del buf1095  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf1110, buf1115, primals_167, primals_163, buf1120, buf1145, 16384, grid=grid(16384), stream=stream0)
        del primals_167
        buf1113 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1118 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1119 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1123 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1143 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1144 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_156, layer_input_54, mul_82], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf1110, buf1115, addmm_156, addmm_151, primals_163, primals_164, buf1120, buf1113, buf1118, buf1119, buf1123, buf1143, buf1144, 128, 128, grid=grid(128), stream=stream0)
        del addmm_151
        del addmm_156
        del primals_163
        del primals_164
        buf1116 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1114, (512, 128), (1, 512), 0), view_424, out=buf1116)
        del view_424
        buf1117 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1114, buf1117, 512, 128, grid=grid(512), stream=stream0)
        buf1121 = buf1115; del buf1115  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1120, (128, 128), (128, 1), 0), permute_1419, out=buf1121)
        del permute_1419
        buf1122 = reinterpret_tensor(buf1110, (128, 128), (128, 1), 0); del buf1110  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1120, (128, 128), (1, 128), 0), view_422, out=buf1122)
        del view_422
        # Source Nodes: [], Original ATen: []
        buf1124 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1121, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_39, clone_default_40, clone_default_41, None, alias_default_27, getitem_142, getitem_143, getitem_144, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_27
        del clone_default_39
        del clone_default_40
        del clone_default_41
        del getitem_142
        del getitem_143
        del getitem_144
        buf1125 = buf1124[0]
        buf1126 = buf1124[1]
        buf1127 = buf1124[2]
        del buf1124
        buf1128 = reinterpret_tensor(buf1114, (128, 512), (512, 1), 0); del buf1114  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1127, (128, 128), (128, 1), 0), permute_1432, out=buf1128)
        del permute_1432
        buf1129 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1127, (128, 128), (1, 128), 0), view_402, out=buf1129)
        buf1130 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1127, buf1130, 128, 128, grid=grid(128), stream=stream0)
        buf1131 = reinterpret_tensor(buf1127, (128, 128), (128, 1), 0); del buf1127  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1126, (128, 128), (128, 1), 0), permute_1436, out=buf1131)
        del permute_1436
        buf1132 = buf1121; del buf1121  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1126, (128, 128), (1, 128), 0), view_406, out=buf1132)
        buf1133 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1126, buf1133, 128, 128, grid=grid(128), stream=stream0)
        buf1134 = reinterpret_tensor(buf1126, (128, 128), (128, 1), 0); del buf1126  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1125, (128, 128), (128, 1), 0), permute_1440, out=buf1134)
        del permute_1440
        buf1135 = reinterpret_tensor(buf1120, (128, 128), (128, 1), 0); del buf1120  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1125, (128, 128), (1, 128), 0), view_406, out=buf1135)
        del view_406
        buf1136 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1125, buf1136, 128, 128, grid=grid(128), stream=stream0)
        del buf1125
        buf1137 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1138 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf1131, buf1134, addmm_152, buf1137, buf1138, 128, 128, grid=grid(128), stream=stream0)
        del addmm_152
        buf1139 = reinterpret_tensor(buf1131, (1, 128, 128), (16384, 128, 1), 0); del buf1131  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1139, buf1134, primals_165, 16384, grid=grid(16384), stream=stream0)
        del primals_165
        buf1140 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1139, (128, 128), (128, 1), 0), permute_1444, out=buf1140)
        del permute_1444
        buf1141 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1139, (128, 128), (1, 128), 0), view_402, out=buf1141)
        buf1142 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1139, buf1142, 128, 128, grid=grid(128), stream=stream0)
        buf1146 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1145, (128, 128), (128, 1), 0), permute_1448, out=buf1146)
        del permute_1448
        buf1147 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1145, (128, 128), (1, 128), 0), view_402, out=buf1147)
        del view_402
        buf1148 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1145, buf1148, 128, 128, grid=grid(128), stream=stream0)
        buf1149 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1151 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_151, mul_73, value_tensor_9], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_24.run(buf1073, buf1128, buf1140, buf1146, addmm_150, buf21, primals_145, primals_146, buf1149, buf1151, 512, 128, grid=grid(512), stream=stream0)
        del addmm_150
        del primals_146
        buf1150 = buf1073; del buf1073  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf1150, buf1128, buf1140, buf1146, primals_161, 65536, grid=grid(65536), stream=stream0)
        del primals_161
        buf1152 = reinterpret_tensor(buf1145, (128, 128), (128, 1), 0); del buf1145  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1150, (128, 512), (512, 1), 0), permute_1452, out=buf1152)
        del permute_1452
        buf1153 = reinterpret_tensor(buf1146, (512, 128), (128, 1), 0); del buf1146  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1150, (512, 128), (1, 512), 0), view_400, out=buf1153)
        del view_400
        buf1154 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf1150, buf1154, 512, 128, grid=grid(512), stream=stream0)
        buf1157 = buf1139; del buf1139  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf1152, primals_159, buf1157, 16384, grid=grid(16384), stream=stream0)
        del primals_159
        buf1158 = buf1140; del buf1140  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1157, (128, 128), (128, 1), 0), permute_1456, out=buf1158)
        del permute_1456
        buf1161 = reinterpret_tensor(buf1158, (1, 128, 512), (65536, 512, 1), 0); del buf1158  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1161, le_56, 65536, grid=grid(65536), stream=stream0)
        del le_56
        buf1162 = buf1134; del buf1134  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1161, (128, 512), (512, 1), 0), permute_1460, out=buf1162)
        del permute_1460
        buf1155 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1156 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1160 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1165 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1166 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_149, attention_output_48, mul_79], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf1152, addmm_149, buf23, primals_157, primals_158, buf1157, buf1162, buf1155, buf1156, buf1160, buf1165, buf1166, 128, 128, grid=grid(128), stream=stream0)
        del addmm_149
        del primals_158
        buf1159 = buf1128; del buf1128  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1157, (128, 128), (1, 128), 0), view_398, out=buf1159)
        del view_398
        buf1163 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1161, (512, 128), (1, 512), 0), view_396, out=buf1163)
        del view_396
        buf1164 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1161, buf1164, 512, 128, grid=grid(512), stream=stream0)
        buf1167 = buf1157; del buf1157  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1167, buf1162, primals_157, 16384, grid=grid(16384), stream=stream0)
        del primals_157
        buf1168 = reinterpret_tensor(buf1161, (128, 512), (512, 1), 0); del buf1161  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1167, (128, 128), (128, 1), 0), permute_1464, out=buf1168)
        del permute_1464
        buf1169 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1167, (128, 128), (1, 128), 0), view_394, out=buf1169)
        del view_394
        buf1171 = reinterpret_tensor(buf1168, (1, 128, 512), (65536, 512, 1), 0); del buf1168  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1171, le_57, 65536, grid=grid(65536), stream=stream0)
        del le_57
        buf1172 = buf1162; del buf1162  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1171, (128, 512), (512, 1), 0), permute_1468, out=buf1172)
        del permute_1468
        buf1177 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf1167, buf1172, primals_155, buf1177, 16384, grid=grid(16384), stream=stream0)
        del primals_155
        buf1178 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1177, (128, 128), (128, 1), 0), permute_1472, out=buf1178)
        del permute_1472
        buf1181 = reinterpret_tensor(buf1178, (1, 128, 512), (65536, 512, 1), 0); del buf1178  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1181, le_58, 65536, grid=grid(65536), stream=stream0)
        del le_58
        buf1182 = buf1152; del buf1152  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1181, (128, 512), (512, 1), 0), permute_1476, out=buf1182)
        del permute_1476
        buf1170 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1175 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1176 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1180 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1185 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1186 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_145, attention_output_46, mul_77], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf1167, buf1172, addmm_145, buf22, primals_153, primals_154, buf1177, buf1182, buf1170, buf1175, buf1176, buf1180, buf1185, buf1186, 128, 128, grid=grid(128), stream=stream0)
        del addmm_145
        del buf1167
        del primals_154
        buf1173 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1171, (512, 128), (1, 512), 0), view_392, out=buf1173)
        del view_392
        buf1174 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1171, buf1174, 512, 128, grid=grid(512), stream=stream0)
        buf1179 = reinterpret_tensor(buf1171, (128, 512), (512, 1), 0); del buf1171  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1177, (128, 128), (1, 128), 0), view_390, out=buf1179)
        del view_390
        buf1183 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1181, (512, 128), (1, 512), 0), view_388, out=buf1183)
        del view_388
        buf1184 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1181, buf1184, 512, 128, grid=grid(512), stream=stream0)
        buf1187 = buf1177; del buf1177  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1187, buf1182, primals_153, 16384, grid=grid(16384), stream=stream0)
        del primals_153
        buf1188 = reinterpret_tensor(buf1181, (128, 512), (512, 1), 0); del buf1181  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1187, (128, 128), (128, 1), 0), permute_1480, out=buf1188)
        del permute_1480
        buf1189 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1187, (128, 128), (1, 128), 0), view_386, out=buf1189)
        del view_386
        buf1191 = reinterpret_tensor(buf1188, (1, 128, 512), (65536, 512, 1), 0); del buf1188  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1191, le_59, 65536, grid=grid(65536), stream=stream0)
        del le_59
        buf1192 = buf1182; del buf1182  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1191, (128, 512), (512, 1), 0), permute_1484, out=buf1192)
        del permute_1484
        buf1197 = buf22; del buf22  # reuse
        buf1222 = reinterpret_tensor(buf1172, (1, 128, 128), (16384, 128, 1), 0); del buf1172  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf1187, buf1192, primals_151, primals_147, buf1197, buf1222, 16384, grid=grid(16384), stream=stream0)
        del primals_151
        buf1190 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1195 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1196 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1200 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1220 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1221 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_141, layer_input_49, mul_74], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf1187, buf1192, addmm_141, addmm_136, primals_147, primals_148, buf1197, buf1190, buf1195, buf1196, buf1200, buf1220, buf1221, 128, 128, grid=grid(128), stream=stream0)
        del addmm_136
        del addmm_141
        del primals_147
        del primals_148
        buf1193 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1191, (512, 128), (1, 512), 0), view_384, out=buf1193)
        del view_384
        buf1194 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1191, buf1194, 512, 128, grid=grid(512), stream=stream0)
        buf1198 = buf1192; del buf1192  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1197, (128, 128), (128, 1), 0), permute_1488, out=buf1198)
        del permute_1488
        buf1199 = reinterpret_tensor(buf1187, (128, 128), (128, 1), 0); del buf1187  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1197, (128, 128), (1, 128), 0), view_382, out=buf1199)
        del view_382
        # Source Nodes: [], Original ATen: []
        buf1201 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1198, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_42, clone_default_43, clone_default_44, None, alias_default_29, getitem_149, getitem_150, getitem_151, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_29
        del clone_default_42
        del clone_default_43
        del clone_default_44
        del getitem_149
        del getitem_150
        del getitem_151
        buf1202 = buf1201[0]
        buf1203 = buf1201[1]
        buf1204 = buf1201[2]
        del buf1201
        buf1205 = reinterpret_tensor(buf1191, (128, 512), (512, 1), 0); del buf1191  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1204, (128, 128), (128, 1), 0), permute_1501, out=buf1205)
        del permute_1501
        buf1206 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1204, (128, 128), (1, 128), 0), view_362, out=buf1206)
        buf1207 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1204, buf1207, 128, 128, grid=grid(128), stream=stream0)
        buf1208 = reinterpret_tensor(buf1204, (128, 128), (128, 1), 0); del buf1204  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1203, (128, 128), (128, 1), 0), permute_1505, out=buf1208)
        del permute_1505
        buf1209 = buf1198; del buf1198  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1203, (128, 128), (1, 128), 0), view_366, out=buf1209)
        buf1210 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1203, buf1210, 128, 128, grid=grid(128), stream=stream0)
        buf1211 = reinterpret_tensor(buf1203, (128, 128), (128, 1), 0); del buf1203  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1202, (128, 128), (128, 1), 0), permute_1509, out=buf1211)
        del permute_1509
        buf1212 = reinterpret_tensor(buf1197, (128, 128), (128, 1), 0); del buf1197  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1202, (128, 128), (1, 128), 0), view_366, out=buf1212)
        del view_366
        buf1213 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1202, buf1213, 128, 128, grid=grid(128), stream=stream0)
        del buf1202
        buf1214 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1215 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf1208, buf1211, addmm_137, buf1214, buf1215, 128, 128, grid=grid(128), stream=stream0)
        del addmm_137
        buf1216 = reinterpret_tensor(buf1208, (1, 128, 128), (16384, 128, 1), 0); del buf1208  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1216, buf1211, primals_149, 16384, grid=grid(16384), stream=stream0)
        del primals_149
        buf1217 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1216, (128, 128), (128, 1), 0), permute_1513, out=buf1217)
        del permute_1513
        buf1218 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1216, (128, 128), (1, 128), 0), view_362, out=buf1218)
        buf1219 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1216, buf1219, 128, 128, grid=grid(128), stream=stream0)
        buf1223 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1222, (128, 128), (128, 1), 0), permute_1517, out=buf1223)
        del permute_1517
        buf1224 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1222, (128, 128), (1, 128), 0), view_362, out=buf1224)
        del view_362
        buf1225 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1222, buf1225, 128, 128, grid=grid(128), stream=stream0)
        buf1226 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1228 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_22.run(buf1150, buf1205, buf1217, buf1223, buf21, buf1226, buf1228, 512, 128, grid=grid(512), stream=stream0)
        buf1227 = buf1150; del buf1150  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf1227, buf1205, buf1217, buf1223, primals_145, 65536, grid=grid(65536), stream=stream0)
        del primals_145
        buf1229 = reinterpret_tensor(buf1222, (128, 128), (128, 1), 0); del buf1222  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1227, (128, 512), (512, 1), 0), permute_1521, out=buf1229)
        del permute_1521
        buf1230 = reinterpret_tensor(buf1223, (512, 128), (128, 1), 0); del buf1223  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1227, (512, 128), (1, 512), 0), view_360, out=buf1230)
        del view_360
        buf1231 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf1227, buf1231, 512, 128, grid=grid(512), stream=stream0)
        buf1234 = buf1216; del buf1216  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf1229, primals_143, buf1234, 16384, grid=grid(16384), stream=stream0)
        del primals_143
        buf1235 = buf1217; del buf1217  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1234, (128, 128), (128, 1), 0), permute_1525, out=buf1235)
        del permute_1525
        buf1238 = reinterpret_tensor(buf1235, (1, 128, 512), (65536, 512, 1), 0); del buf1235  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1238, le_60, 65536, grid=grid(65536), stream=stream0)
        del le_60
        buf1239 = buf1211; del buf1211  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1238, (128, 512), (512, 1), 0), permute_1529, out=buf1239)
        del permute_1529
        buf1232 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1233 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1237 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1242 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1243 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_134, attention_output_43, mul_71], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf1229, addmm_134, buf20, primals_141, primals_142, buf1234, buf1239, buf1232, buf1233, buf1237, buf1242, buf1243, 128, 128, grid=grid(128), stream=stream0)
        del addmm_134
        del primals_142
        buf1236 = buf1205; del buf1205  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1234, (128, 128), (1, 128), 0), view_358, out=buf1236)
        del view_358
        buf1240 = reinterpret_tensor(buf21, (512, 128), (128, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1238, (512, 128), (1, 512), 0), view_356, out=buf1240)
        del view_356
        buf1241 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1238, buf1241, 512, 128, grid=grid(512), stream=stream0)
        buf1244 = buf1234; del buf1234  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1244, buf1239, primals_141, 16384, grid=grid(16384), stream=stream0)
        del primals_141
        buf1245 = reinterpret_tensor(buf1238, (128, 512), (512, 1), 0); del buf1238  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1244, (128, 128), (128, 1), 0), permute_1533, out=buf1245)
        del permute_1533
        buf1246 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1244, (128, 128), (1, 128), 0), view_354, out=buf1246)
        del view_354
        buf1248 = reinterpret_tensor(buf1245, (1, 128, 512), (65536, 512, 1), 0); del buf1245  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1248, le_61, 65536, grid=grid(65536), stream=stream0)
        del le_61
        buf1249 = buf1239; del buf1239  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1248, (128, 512), (512, 1), 0), permute_1537, out=buf1249)
        del permute_1537
        buf1254 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf1244, buf1249, primals_139, buf1254, 16384, grid=grid(16384), stream=stream0)
        del primals_139
        buf1255 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1254, (128, 128), (128, 1), 0), permute_1541, out=buf1255)
        del permute_1541
        buf1258 = reinterpret_tensor(buf1255, (1, 128, 512), (65536, 512, 1), 0); del buf1255  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1258, le_62, 65536, grid=grid(65536), stream=stream0)
        del le_62
        buf1259 = buf1229; del buf1229  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1258, (128, 512), (512, 1), 0), permute_1545, out=buf1259)
        del permute_1545
        buf1247 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1252 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1253 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1257 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1262 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1263 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_130, attention_output_41, mul_69], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf1244, buf1249, addmm_130, buf19, primals_137, primals_138, buf1254, buf1259, buf1247, buf1252, buf1253, buf1257, buf1262, buf1263, 128, 128, grid=grid(128), stream=stream0)
        del addmm_130
        del buf1244
        del primals_138
        buf1250 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1248, (512, 128), (1, 512), 0), view_352, out=buf1250)
        del view_352
        buf1251 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1248, buf1251, 512, 128, grid=grid(512), stream=stream0)
        buf1256 = reinterpret_tensor(buf1248, (128, 512), (512, 1), 0); del buf1248  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1254, (128, 128), (1, 128), 0), view_350, out=buf1256)
        del view_350
        buf1260 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1258, (512, 128), (1, 512), 0), view_348, out=buf1260)
        del view_348
        buf1261 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1258, buf1261, 512, 128, grid=grid(512), stream=stream0)
        buf1264 = buf1254; del buf1254  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1264, buf1259, primals_137, 16384, grid=grid(16384), stream=stream0)
        del primals_137
        buf1265 = reinterpret_tensor(buf1258, (128, 512), (512, 1), 0); del buf1258  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1264, (128, 128), (128, 1), 0), permute_1549, out=buf1265)
        del permute_1549
        buf1266 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1264, (128, 128), (1, 128), 0), view_346, out=buf1266)
        del view_346
        buf1268 = reinterpret_tensor(buf1265, (1, 128, 512), (65536, 512, 1), 0); del buf1265  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1268, le_63, 65536, grid=grid(65536), stream=stream0)
        del le_63
        buf1269 = buf1259; del buf1259  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1268, (128, 512), (512, 1), 0), permute_1553, out=buf1269)
        del permute_1553
        buf1274 = buf19; del buf19  # reuse
        buf1299 = reinterpret_tensor(buf1249, (1, 128, 128), (16384, 128, 1), 0); del buf1249  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf1264, buf1269, primals_135, primals_131, buf1274, buf1299, 16384, grid=grid(16384), stream=stream0)
        del primals_135
        buf1267 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1272 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1273 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1277 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1297 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1298 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_126, layer_input_44, mul_66], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf1264, buf1269, addmm_126, addmm_121, primals_131, primals_132, buf1274, buf1267, buf1272, buf1273, buf1277, buf1297, buf1298, 128, 128, grid=grid(128), stream=stream0)
        del addmm_121
        del addmm_126
        del primals_131
        del primals_132
        buf1270 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1268, (512, 128), (1, 512), 0), view_344, out=buf1270)
        del view_344
        buf1271 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1268, buf1271, 512, 128, grid=grid(512), stream=stream0)
        buf1275 = buf1269; del buf1269  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1274, (128, 128), (128, 1), 0), permute_1557, out=buf1275)
        del permute_1557
        buf1276 = reinterpret_tensor(buf1264, (128, 128), (128, 1), 0); del buf1264  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1274, (128, 128), (1, 128), 0), view_342, out=buf1276)
        del view_342
        # Source Nodes: [], Original ATen: []
        buf1278 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1275, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_45, clone_default_46, clone_default_47, None, alias_default_31, getitem_156, getitem_157, getitem_158, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_31
        del clone_default_45
        del clone_default_46
        del clone_default_47
        del getitem_156
        del getitem_157
        del getitem_158
        buf1279 = buf1278[0]
        buf1280 = buf1278[1]
        buf1281 = buf1278[2]
        del buf1278
        buf1282 = reinterpret_tensor(buf1268, (128, 512), (512, 1), 0); del buf1268  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1281, (128, 128), (128, 1), 0), permute_1570, out=buf1282)
        del permute_1570
        buf1283 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1281, (128, 128), (1, 128), 0), view_322, out=buf1283)
        buf1284 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1281, buf1284, 128, 128, grid=grid(128), stream=stream0)
        buf1285 = reinterpret_tensor(buf1281, (128, 128), (128, 1), 0); del buf1281  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1280, (128, 128), (128, 1), 0), permute_1574, out=buf1285)
        del permute_1574
        buf1286 = buf1275; del buf1275  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1280, (128, 128), (1, 128), 0), view_326, out=buf1286)
        buf1287 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1280, buf1287, 128, 128, grid=grid(128), stream=stream0)
        buf1288 = reinterpret_tensor(buf1280, (128, 128), (128, 1), 0); del buf1280  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1279, (128, 128), (128, 1), 0), permute_1578, out=buf1288)
        del permute_1578
        buf1289 = reinterpret_tensor(buf1274, (128, 128), (128, 1), 0); del buf1274  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1279, (128, 128), (1, 128), 0), view_326, out=buf1289)
        del view_326
        buf1290 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1279, buf1290, 128, 128, grid=grid(128), stream=stream0)
        del buf1279
        buf1291 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1292 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf1285, buf1288, addmm_122, buf1291, buf1292, 128, 128, grid=grid(128), stream=stream0)
        del addmm_122
        buf1293 = reinterpret_tensor(buf1285, (1, 128, 128), (16384, 128, 1), 0); del buf1285  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1293, buf1288, primals_133, 16384, grid=grid(16384), stream=stream0)
        del primals_133
        buf1294 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1293, (128, 128), (128, 1), 0), permute_1582, out=buf1294)
        del permute_1582
        buf1295 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1293, (128, 128), (1, 128), 0), view_322, out=buf1295)
        buf1296 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1293, buf1296, 128, 128, grid=grid(128), stream=stream0)
        buf1300 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1299, (128, 128), (128, 1), 0), permute_1586, out=buf1300)
        del permute_1586
        buf1301 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1299, (128, 128), (1, 128), 0), view_322, out=buf1301)
        del view_322
        buf1302 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1299, buf1302, 128, 128, grid=grid(128), stream=stream0)
        buf1303 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1305 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_121, mul_57, value_tensor_7], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_24.run(buf1227, buf1282, buf1294, buf1300, addmm_120, buf16, primals_113, primals_114, buf1303, buf1305, 512, 128, grid=grid(512), stream=stream0)
        del addmm_120
        del primals_114
        buf1304 = buf1227; del buf1227  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf1304, buf1282, buf1294, buf1300, primals_129, 65536, grid=grid(65536), stream=stream0)
        del primals_129
        buf1306 = reinterpret_tensor(buf1299, (128, 128), (128, 1), 0); del buf1299  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1304, (128, 512), (512, 1), 0), permute_1590, out=buf1306)
        del permute_1590
        buf1307 = reinterpret_tensor(buf1300, (512, 128), (128, 1), 0); del buf1300  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1304, (512, 128), (1, 512), 0), view_320, out=buf1307)
        del view_320
        buf1308 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf1304, buf1308, 512, 128, grid=grid(512), stream=stream0)
        buf1311 = buf1293; del buf1293  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf1306, primals_127, buf1311, 16384, grid=grid(16384), stream=stream0)
        del primals_127
        buf1312 = buf1294; del buf1294  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1311, (128, 128), (128, 1), 0), permute_1594, out=buf1312)
        del permute_1594
        buf1315 = reinterpret_tensor(buf1312, (1, 128, 512), (65536, 512, 1), 0); del buf1312  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1315, le_64, 65536, grid=grid(65536), stream=stream0)
        del le_64
        buf1316 = buf1288; del buf1288  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1315, (128, 512), (512, 1), 0), permute_1598, out=buf1316)
        del permute_1598
        buf1309 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1310 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1314 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1319 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1320 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_119, attention_output_38, mul_63], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf1306, addmm_119, buf18, primals_125, primals_126, buf1311, buf1316, buf1309, buf1310, buf1314, buf1319, buf1320, 128, 128, grid=grid(128), stream=stream0)
        del addmm_119
        del primals_126
        buf1313 = buf1282; del buf1282  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1311, (128, 128), (1, 128), 0), view_318, out=buf1313)
        del view_318
        buf1317 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1315, (512, 128), (1, 512), 0), view_316, out=buf1317)
        del view_316
        buf1318 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1315, buf1318, 512, 128, grid=grid(512), stream=stream0)
        buf1321 = buf1311; del buf1311  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1321, buf1316, primals_125, 16384, grid=grid(16384), stream=stream0)
        del primals_125
        buf1322 = reinterpret_tensor(buf1315, (128, 512), (512, 1), 0); del buf1315  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1321, (128, 128), (128, 1), 0), permute_1602, out=buf1322)
        del permute_1602
        buf1323 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1321, (128, 128), (1, 128), 0), view_314, out=buf1323)
        del view_314
        buf1325 = reinterpret_tensor(buf1322, (1, 128, 512), (65536, 512, 1), 0); del buf1322  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1325, le_65, 65536, grid=grid(65536), stream=stream0)
        del le_65
        buf1326 = buf1316; del buf1316  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1325, (128, 512), (512, 1), 0), permute_1606, out=buf1326)
        del permute_1606
        buf1331 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf1321, buf1326, primals_123, buf1331, 16384, grid=grid(16384), stream=stream0)
        del primals_123
        buf1332 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1331, (128, 128), (128, 1), 0), permute_1610, out=buf1332)
        del permute_1610
        buf1335 = reinterpret_tensor(buf1332, (1, 128, 512), (65536, 512, 1), 0); del buf1332  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1335, le_66, 65536, grid=grid(65536), stream=stream0)
        del le_66
        buf1336 = buf1306; del buf1306  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1335, (128, 512), (512, 1), 0), permute_1614, out=buf1336)
        del permute_1614
        buf1324 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1329 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1330 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1334 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1339 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1340 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_115, attention_output_36, mul_61], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf1321, buf1326, addmm_115, buf17, primals_121, primals_122, buf1331, buf1336, buf1324, buf1329, buf1330, buf1334, buf1339, buf1340, 128, 128, grid=grid(128), stream=stream0)
        del addmm_115
        del buf1321
        del primals_122
        buf1327 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1325, (512, 128), (1, 512), 0), view_312, out=buf1327)
        del view_312
        buf1328 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1325, buf1328, 512, 128, grid=grid(512), stream=stream0)
        buf1333 = reinterpret_tensor(buf1325, (128, 512), (512, 1), 0); del buf1325  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1331, (128, 128), (1, 128), 0), view_310, out=buf1333)
        del view_310
        buf1337 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1335, (512, 128), (1, 512), 0), view_308, out=buf1337)
        del view_308
        buf1338 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1335, buf1338, 512, 128, grid=grid(512), stream=stream0)
        buf1341 = buf1331; del buf1331  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1341, buf1336, primals_121, 16384, grid=grid(16384), stream=stream0)
        del primals_121
        buf1342 = reinterpret_tensor(buf1335, (128, 512), (512, 1), 0); del buf1335  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1341, (128, 128), (128, 1), 0), permute_1618, out=buf1342)
        del permute_1618
        buf1343 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1341, (128, 128), (1, 128), 0), view_306, out=buf1343)
        del view_306
        buf1345 = reinterpret_tensor(buf1342, (1, 128, 512), (65536, 512, 1), 0); del buf1342  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1345, le_67, 65536, grid=grid(65536), stream=stream0)
        del le_67
        buf1346 = buf1336; del buf1336  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1345, (128, 512), (512, 1), 0), permute_1622, out=buf1346)
        del permute_1622
        buf1351 = buf17; del buf17  # reuse
        buf1376 = reinterpret_tensor(buf1326, (1, 128, 128), (16384, 128, 1), 0); del buf1326  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf1341, buf1346, primals_119, primals_115, buf1351, buf1376, 16384, grid=grid(16384), stream=stream0)
        del primals_119
        buf1344 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1349 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1350 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1354 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1374 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1375 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_111, layer_input_39, mul_58], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf1341, buf1346, addmm_111, addmm_106, primals_115, primals_116, buf1351, buf1344, buf1349, buf1350, buf1354, buf1374, buf1375, 128, 128, grid=grid(128), stream=stream0)
        del addmm_106
        del addmm_111
        del primals_115
        del primals_116
        buf1347 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1345, (512, 128), (1, 512), 0), view_304, out=buf1347)
        del view_304
        buf1348 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1345, buf1348, 512, 128, grid=grid(512), stream=stream0)
        buf1352 = buf1346; del buf1346  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1351, (128, 128), (128, 1), 0), permute_1626, out=buf1352)
        del permute_1626
        buf1353 = reinterpret_tensor(buf1341, (128, 128), (128, 1), 0); del buf1341  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1351, (128, 128), (1, 128), 0), view_302, out=buf1353)
        del view_302
        # Source Nodes: [], Original ATen: []
        buf1355 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1352, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_48, clone_default_49, clone_default_50, None, alias_default_33, getitem_163, getitem_164, getitem_165, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_33
        del clone_default_48
        del clone_default_49
        del clone_default_50
        del getitem_163
        del getitem_164
        del getitem_165
        buf1356 = buf1355[0]
        buf1357 = buf1355[1]
        buf1358 = buf1355[2]
        del buf1355
        buf1359 = reinterpret_tensor(buf1345, (128, 512), (512, 1), 0); del buf1345  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1358, (128, 128), (128, 1), 0), permute_1639, out=buf1359)
        del permute_1639
        buf1360 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1358, (128, 128), (1, 128), 0), view_282, out=buf1360)
        buf1361 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1358, buf1361, 128, 128, grid=grid(128), stream=stream0)
        buf1362 = reinterpret_tensor(buf1358, (128, 128), (128, 1), 0); del buf1358  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1357, (128, 128), (128, 1), 0), permute_1643, out=buf1362)
        del permute_1643
        buf1363 = buf1352; del buf1352  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1357, (128, 128), (1, 128), 0), view_286, out=buf1363)
        buf1364 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1357, buf1364, 128, 128, grid=grid(128), stream=stream0)
        buf1365 = reinterpret_tensor(buf1357, (128, 128), (128, 1), 0); del buf1357  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1356, (128, 128), (128, 1), 0), permute_1647, out=buf1365)
        del permute_1647
        buf1366 = reinterpret_tensor(buf1351, (128, 128), (128, 1), 0); del buf1351  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1356, (128, 128), (1, 128), 0), view_286, out=buf1366)
        del view_286
        buf1367 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1356, buf1367, 128, 128, grid=grid(128), stream=stream0)
        del buf1356
        buf1368 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1369 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf1362, buf1365, addmm_107, buf1368, buf1369, 128, 128, grid=grid(128), stream=stream0)
        del addmm_107
        buf1370 = reinterpret_tensor(buf1362, (1, 128, 128), (16384, 128, 1), 0); del buf1362  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1370, buf1365, primals_117, 16384, grid=grid(16384), stream=stream0)
        del primals_117
        buf1371 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1370, (128, 128), (128, 1), 0), permute_1651, out=buf1371)
        del permute_1651
        buf1372 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1370, (128, 128), (1, 128), 0), view_282, out=buf1372)
        buf1373 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1370, buf1373, 128, 128, grid=grid(128), stream=stream0)
        buf1377 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1376, (128, 128), (128, 1), 0), permute_1655, out=buf1377)
        del permute_1655
        buf1378 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1376, (128, 128), (1, 128), 0), view_282, out=buf1378)
        del view_282
        buf1379 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1376, buf1379, 128, 128, grid=grid(128), stream=stream0)
        buf1380 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1382 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_22.run(buf1304, buf1359, buf1371, buf1377, buf16, buf1380, buf1382, 512, 128, grid=grid(512), stream=stream0)
        buf1381 = buf1304; del buf1304  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf1381, buf1359, buf1371, buf1377, primals_113, 65536, grid=grid(65536), stream=stream0)
        del primals_113
        buf1383 = reinterpret_tensor(buf1376, (128, 128), (128, 1), 0); del buf1376  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1381, (128, 512), (512, 1), 0), permute_1659, out=buf1383)
        del permute_1659
        buf1384 = reinterpret_tensor(buf1377, (512, 128), (128, 1), 0); del buf1377  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1381, (512, 128), (1, 512), 0), view_280, out=buf1384)
        del view_280
        buf1385 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf1381, buf1385, 512, 128, grid=grid(512), stream=stream0)
        buf1388 = buf1370; del buf1370  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf1383, primals_111, buf1388, 16384, grid=grid(16384), stream=stream0)
        del primals_111
        buf1389 = buf1371; del buf1371  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1388, (128, 128), (128, 1), 0), permute_1663, out=buf1389)
        del permute_1663
        buf1392 = reinterpret_tensor(buf1389, (1, 128, 512), (65536, 512, 1), 0); del buf1389  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1392, le_68, 65536, grid=grid(65536), stream=stream0)
        del le_68
        buf1393 = buf1365; del buf1365  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1392, (128, 512), (512, 1), 0), permute_1667, out=buf1393)
        del permute_1667
        buf1386 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1387 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1391 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1396 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1397 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_104, attention_output_33, mul_55], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf1383, addmm_104, buf15, primals_109, primals_110, buf1388, buf1393, buf1386, buf1387, buf1391, buf1396, buf1397, 128, 128, grid=grid(128), stream=stream0)
        del addmm_104
        del primals_110
        buf1390 = buf1359; del buf1359  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1388, (128, 128), (1, 128), 0), view_278, out=buf1390)
        del view_278
        buf1394 = reinterpret_tensor(buf16, (512, 128), (128, 1), 0); del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1392, (512, 128), (1, 512), 0), view_276, out=buf1394)
        del view_276
        buf1395 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1392, buf1395, 512, 128, grid=grid(512), stream=stream0)
        buf1398 = buf1388; del buf1388  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1398, buf1393, primals_109, 16384, grid=grid(16384), stream=stream0)
        del primals_109
        buf1399 = reinterpret_tensor(buf1392, (128, 512), (512, 1), 0); del buf1392  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1398, (128, 128), (128, 1), 0), permute_1671, out=buf1399)
        del permute_1671
        buf1400 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1398, (128, 128), (1, 128), 0), view_274, out=buf1400)
        del view_274
        buf1402 = reinterpret_tensor(buf1399, (1, 128, 512), (65536, 512, 1), 0); del buf1399  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1402, le_69, 65536, grid=grid(65536), stream=stream0)
        del le_69
        buf1403 = buf1393; del buf1393  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1402, (128, 512), (512, 1), 0), permute_1675, out=buf1403)
        del permute_1675
        buf1408 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf1398, buf1403, primals_107, buf1408, 16384, grid=grid(16384), stream=stream0)
        del primals_107
        buf1409 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1408, (128, 128), (128, 1), 0), permute_1679, out=buf1409)
        del permute_1679
        buf1412 = reinterpret_tensor(buf1409, (1, 128, 512), (65536, 512, 1), 0); del buf1409  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1412, le_70, 65536, grid=grid(65536), stream=stream0)
        del le_70
        buf1413 = buf1383; del buf1383  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1412, (128, 512), (512, 1), 0), permute_1683, out=buf1413)
        del permute_1683
        buf1401 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1406 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1407 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1411 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1416 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1417 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_100, attention_output_31, mul_53], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf1398, buf1403, addmm_100, buf14, primals_105, primals_106, buf1408, buf1413, buf1401, buf1406, buf1407, buf1411, buf1416, buf1417, 128, 128, grid=grid(128), stream=stream0)
        del addmm_100
        del buf1398
        del primals_106
        buf1404 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1402, (512, 128), (1, 512), 0), view_272, out=buf1404)
        del view_272
        buf1405 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1402, buf1405, 512, 128, grid=grid(512), stream=stream0)
        buf1410 = reinterpret_tensor(buf1402, (128, 512), (512, 1), 0); del buf1402  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1408, (128, 128), (1, 128), 0), view_270, out=buf1410)
        del view_270
        buf1414 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1412, (512, 128), (1, 512), 0), view_268, out=buf1414)
        del view_268
        buf1415 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1412, buf1415, 512, 128, grid=grid(512), stream=stream0)
        buf1418 = buf1408; del buf1408  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1418, buf1413, primals_105, 16384, grid=grid(16384), stream=stream0)
        del primals_105
        buf1419 = reinterpret_tensor(buf1412, (128, 512), (512, 1), 0); del buf1412  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1418, (128, 128), (128, 1), 0), permute_1687, out=buf1419)
        del permute_1687
        buf1420 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1418, (128, 128), (1, 128), 0), view_266, out=buf1420)
        del view_266
        buf1422 = reinterpret_tensor(buf1419, (1, 128, 512), (65536, 512, 1), 0); del buf1419  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1422, le_71, 65536, grid=grid(65536), stream=stream0)
        del le_71
        buf1423 = buf1413; del buf1413  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1422, (128, 512), (512, 1), 0), permute_1691, out=buf1423)
        del permute_1691
        buf1428 = reinterpret_tensor(buf1403, (1, 128, 128), (16384, 128, 1), 0); del buf1403  # reuse
        buf1453 = buf14; del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf1418, buf1423, primals_103, primals_99, buf1428, buf1453, 16384, grid=grid(16384), stream=stream0)
        del primals_103
        buf1421 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1426 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1427 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1431 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1451 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1452 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_96, layer_input_34, mul_50], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf1418, buf1423, addmm_96, addmm_91, primals_99, primals_100, buf1428, buf1421, buf1426, buf1427, buf1431, buf1451, buf1452, 128, 128, grid=grid(128), stream=stream0)
        del addmm_91
        del addmm_96
        del primals_100
        del primals_99
        buf1424 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1422, (512, 128), (1, 512), 0), view_264, out=buf1424)
        del view_264
        buf1425 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1422, buf1425, 512, 128, grid=grid(512), stream=stream0)
        buf1429 = buf1423; del buf1423  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1428, (128, 128), (128, 1), 0), permute_1695, out=buf1429)
        del permute_1695
        buf1430 = reinterpret_tensor(buf1418, (128, 128), (128, 1), 0); del buf1418  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1428, (128, 128), (1, 128), 0), view_262, out=buf1430)
        del view_262
        # Source Nodes: [], Original ATen: []
        buf1432 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1429, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_51, clone_default_52, clone_default_53, None, alias_default_35, getitem_170, getitem_171, getitem_172, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_35
        del clone_default_51
        del clone_default_52
        del clone_default_53
        del getitem_170
        del getitem_171
        del getitem_172
        buf1433 = buf1432[0]
        buf1434 = buf1432[1]
        buf1435 = buf1432[2]
        del buf1432
        buf1436 = reinterpret_tensor(buf1422, (128, 512), (512, 1), 0); del buf1422  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1435, (128, 128), (128, 1), 0), permute_1708, out=buf1436)
        del permute_1708
        buf1437 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1435, (128, 128), (1, 128), 0), view_242, out=buf1437)
        buf1438 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1435, buf1438, 128, 128, grid=grid(128), stream=stream0)
        buf1439 = reinterpret_tensor(buf1435, (128, 128), (128, 1), 0); del buf1435  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1434, (128, 128), (128, 1), 0), permute_1712, out=buf1439)
        del permute_1712
        buf1440 = buf1429; del buf1429  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1434, (128, 128), (1, 128), 0), view_246, out=buf1440)
        buf1441 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1434, buf1441, 128, 128, grid=grid(128), stream=stream0)
        buf1442 = reinterpret_tensor(buf1434, (128, 128), (128, 1), 0); del buf1434  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1433, (128, 128), (128, 1), 0), permute_1716, out=buf1442)
        del permute_1716
        buf1443 = reinterpret_tensor(buf1428, (128, 128), (128, 1), 0); del buf1428  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1433, (128, 128), (1, 128), 0), view_246, out=buf1443)
        del view_246
        buf1444 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1433, buf1444, 128, 128, grid=grid(128), stream=stream0)
        del buf1433
        buf1445 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1446 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf1439, buf1442, addmm_92, buf1445, buf1446, 128, 128, grid=grid(128), stream=stream0)
        del addmm_92
        buf1447 = reinterpret_tensor(buf1439, (1, 128, 128), (16384, 128, 1), 0); del buf1439  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1447, buf1442, primals_101, 16384, grid=grid(16384), stream=stream0)
        del primals_101
        buf1448 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1447, (128, 128), (128, 1), 0), permute_1720, out=buf1448)
        del permute_1720
        buf1449 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1447, (128, 128), (1, 128), 0), view_242, out=buf1449)
        buf1450 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1447, buf1450, 128, 128, grid=grid(128), stream=stream0)
        buf1454 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1453, (128, 128), (128, 1), 0), permute_1724, out=buf1454)
        del permute_1724
        buf1455 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1453, (128, 128), (1, 128), 0), view_242, out=buf1455)
        del view_242
        buf1456 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1453, buf1456, 128, 128, grid=grid(128), stream=stream0)
        buf1457 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1459 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_91, mul_41, value_tensor_5], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_24.run(buf1381, buf1436, buf1448, buf1454, addmm_90, buf11, primals_81, primals_82, buf1457, buf1459, 512, 128, grid=grid(512), stream=stream0)
        del addmm_90
        del primals_82
        buf1458 = buf1381; del buf1381  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf1458, buf1436, buf1448, buf1454, primals_97, 65536, grid=grid(65536), stream=stream0)
        del primals_97
        buf1460 = reinterpret_tensor(buf1453, (128, 128), (128, 1), 0); del buf1453  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1458, (128, 512), (512, 1), 0), permute_1728, out=buf1460)
        del permute_1728
        buf1461 = reinterpret_tensor(buf1454, (512, 128), (128, 1), 0); del buf1454  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1458, (512, 128), (1, 512), 0), view_240, out=buf1461)
        del view_240
        buf1462 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf1458, buf1462, 512, 128, grid=grid(512), stream=stream0)
        buf1465 = buf1447; del buf1447  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf1460, primals_95, buf1465, 16384, grid=grid(16384), stream=stream0)
        del primals_95
        buf1466 = buf1448; del buf1448  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1465, (128, 128), (128, 1), 0), permute_1732, out=buf1466)
        del permute_1732
        buf1469 = reinterpret_tensor(buf1466, (1, 128, 512), (65536, 512, 1), 0); del buf1466  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1469, le_72, 65536, grid=grid(65536), stream=stream0)
        del le_72
        buf1470 = buf1442; del buf1442  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1469, (128, 512), (512, 1), 0), permute_1736, out=buf1470)
        del permute_1736
        buf1463 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1464 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1468 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1473 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1474 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_89, attention_output_28, mul_47], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf1460, addmm_89, buf13, primals_93, primals_94, buf1465, buf1470, buf1463, buf1464, buf1468, buf1473, buf1474, 128, 128, grid=grid(128), stream=stream0)
        del addmm_89
        del primals_94
        buf1467 = buf1436; del buf1436  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1465, (128, 128), (1, 128), 0), view_238, out=buf1467)
        del view_238
        buf1471 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1469, (512, 128), (1, 512), 0), view_236, out=buf1471)
        del view_236
        buf1472 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1469, buf1472, 512, 128, grid=grid(512), stream=stream0)
        buf1475 = buf1465; del buf1465  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1475, buf1470, primals_93, 16384, grid=grid(16384), stream=stream0)
        del primals_93
        buf1476 = reinterpret_tensor(buf1469, (128, 512), (512, 1), 0); del buf1469  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1475, (128, 128), (128, 1), 0), permute_1740, out=buf1476)
        del permute_1740
        buf1477 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1475, (128, 128), (1, 128), 0), view_234, out=buf1477)
        del view_234
        buf1479 = reinterpret_tensor(buf1476, (1, 128, 512), (65536, 512, 1), 0); del buf1476  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1479, le_73, 65536, grid=grid(65536), stream=stream0)
        del le_73
        buf1480 = buf1470; del buf1470  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1479, (128, 512), (512, 1), 0), permute_1744, out=buf1480)
        del permute_1744
        buf1485 = reinterpret_tensor(buf1460, (1, 128, 128), (16384, 128, 1), 0); del buf1460  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf1475, buf1480, primals_91, buf1485, 16384, grid=grid(16384), stream=stream0)
        del primals_91
        buf1486 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1485, (128, 128), (128, 1), 0), permute_1748, out=buf1486)
        del permute_1748
        buf1489 = reinterpret_tensor(buf1486, (1, 128, 512), (65536, 512, 1), 0); del buf1486  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1489, le_74, 65536, grid=grid(65536), stream=stream0)
        del le_74
        buf1490 = reinterpret_tensor(buf13, (128, 128), (128, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1489, (128, 512), (512, 1), 0), permute_1752, out=buf1490)
        del permute_1752
        buf1478 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1483 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1484 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1488 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1493 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1494 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_85, attention_output_26, mul_45], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf1475, buf1480, addmm_85, buf12, primals_89, primals_90, buf1485, buf1490, buf1478, buf1483, buf1484, buf1488, buf1493, buf1494, 128, 128, grid=grid(128), stream=stream0)
        del addmm_85
        del buf12
        del primals_90
        buf1481 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1479, (512, 128), (1, 512), 0), view_232, out=buf1481)
        del view_232
        buf1482 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1479, buf1482, 512, 128, grid=grid(512), stream=stream0)
        buf1487 = reinterpret_tensor(buf1479, (128, 512), (512, 1), 0); del buf1479  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1485, (128, 128), (1, 128), 0), view_230, out=buf1487)
        del view_230
        buf1491 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1489, (512, 128), (1, 512), 0), view_228, out=buf1491)
        del view_228
        buf1492 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1489, buf1492, 512, 128, grid=grid(512), stream=stream0)
        buf1495 = buf1485; del buf1485  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1495, buf1490, primals_89, 16384, grid=grid(16384), stream=stream0)
        del primals_89
        buf1496 = reinterpret_tensor(buf1489, (128, 512), (512, 1), 0); del buf1489  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1495, (128, 128), (128, 1), 0), permute_1756, out=buf1496)
        del permute_1756
        buf1497 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1495, (128, 128), (1, 128), 0), view_226, out=buf1497)
        del view_226
        buf1499 = reinterpret_tensor(buf1496, (1, 128, 512), (65536, 512, 1), 0); del buf1496  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1499, le_75, 65536, grid=grid(65536), stream=stream0)
        del le_75
        buf1500 = buf1490; del buf1490  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1499, (128, 512), (512, 1), 0), permute_1760, out=buf1500)
        del permute_1760
        buf1505 = reinterpret_tensor(buf1480, (1, 128, 128), (16384, 128, 1), 0); del buf1480  # reuse
        buf1530 = buf1475; del buf1475  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf1495, buf1500, primals_87, primals_83, buf1505, buf1530, 16384, grid=grid(16384), stream=stream0)
        del primals_87
        buf1498 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1503 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1504 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1508 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1528 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1529 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_81, layer_input_29, mul_42], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf1495, buf1500, addmm_81, addmm_76, primals_83, primals_84, buf1505, buf1498, buf1503, buf1504, buf1508, buf1528, buf1529, 128, 128, grid=grid(128), stream=stream0)
        del addmm_76
        del addmm_81
        del primals_83
        del primals_84
        buf1501 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1499, (512, 128), (1, 512), 0), view_224, out=buf1501)
        del view_224
        buf1502 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1499, buf1502, 512, 128, grid=grid(512), stream=stream0)
        buf1506 = buf1500; del buf1500  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1505, (128, 128), (128, 1), 0), permute_1764, out=buf1506)
        del permute_1764
        buf1507 = reinterpret_tensor(buf1495, (128, 128), (128, 1), 0); del buf1495  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1505, (128, 128), (1, 128), 0), view_222, out=buf1507)
        del view_222
        # Source Nodes: [], Original ATen: []
        buf1509 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1506, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_54, clone_default_55, clone_default_56, None, alias_default_37, getitem_177, getitem_178, getitem_179, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_37
        del clone_default_54
        del clone_default_55
        del clone_default_56
        del getitem_177
        del getitem_178
        del getitem_179
        buf1510 = buf1509[0]
        buf1511 = buf1509[1]
        buf1512 = buf1509[2]
        del buf1509
        buf1513 = reinterpret_tensor(buf1499, (128, 512), (512, 1), 0); del buf1499  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1512, (128, 128), (128, 1), 0), permute_1777, out=buf1513)
        del permute_1777
        buf1514 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1512, (128, 128), (1, 128), 0), view_202, out=buf1514)
        buf1515 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1512, buf1515, 128, 128, grid=grid(128), stream=stream0)
        buf1516 = reinterpret_tensor(buf1512, (128, 128), (128, 1), 0); del buf1512  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1511, (128, 128), (128, 1), 0), permute_1781, out=buf1516)
        del permute_1781
        buf1517 = buf1506; del buf1506  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1511, (128, 128), (1, 128), 0), view_206, out=buf1517)
        buf1518 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1511, buf1518, 128, 128, grid=grid(128), stream=stream0)
        buf1519 = reinterpret_tensor(buf1511, (128, 128), (128, 1), 0); del buf1511  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1510, (128, 128), (128, 1), 0), permute_1785, out=buf1519)
        del permute_1785
        buf1520 = reinterpret_tensor(buf1505, (128, 128), (128, 1), 0); del buf1505  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1510, (128, 128), (1, 128), 0), view_206, out=buf1520)
        del view_206
        buf1521 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1510, buf1521, 128, 128, grid=grid(128), stream=stream0)
        del buf1510
        buf1522 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1523 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf1516, buf1519, addmm_77, buf1522, buf1523, 128, 128, grid=grid(128), stream=stream0)
        del addmm_77
        buf1524 = reinterpret_tensor(buf1516, (1, 128, 128), (16384, 128, 1), 0); del buf1516  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1524, buf1519, primals_85, 16384, grid=grid(16384), stream=stream0)
        del primals_85
        buf1525 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1524, (128, 128), (128, 1), 0), permute_1789, out=buf1525)
        del permute_1789
        buf1526 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1524, (128, 128), (1, 128), 0), view_202, out=buf1526)
        buf1527 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1524, buf1527, 128, 128, grid=grid(128), stream=stream0)
        buf1531 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1530, (128, 128), (128, 1), 0), permute_1793, out=buf1531)
        del permute_1793
        buf1532 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1530, (128, 128), (1, 128), 0), view_202, out=buf1532)
        del view_202
        buf1533 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1530, buf1533, 128, 128, grid=grid(128), stream=stream0)
        buf1534 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1536 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_22.run(buf1458, buf1513, buf1525, buf1531, buf11, buf1534, buf1536, 512, 128, grid=grid(512), stream=stream0)
        buf1535 = buf1458; del buf1458  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf1535, buf1513, buf1525, buf1531, primals_81, 65536, grid=grid(65536), stream=stream0)
        del primals_81
        buf1537 = reinterpret_tensor(buf1530, (128, 128), (128, 1), 0); del buf1530  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1535, (128, 512), (512, 1), 0), permute_1797, out=buf1537)
        del permute_1797
        buf1538 = reinterpret_tensor(buf1531, (512, 128), (128, 1), 0); del buf1531  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1535, (512, 128), (1, 512), 0), view_200, out=buf1538)
        del view_200
        buf1539 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf1535, buf1539, 512, 128, grid=grid(512), stream=stream0)
        buf1542 = buf1524; del buf1524  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf1537, primals_79, buf1542, 16384, grid=grid(16384), stream=stream0)
        del primals_79
        buf1543 = buf1525; del buf1525  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1542, (128, 128), (128, 1), 0), permute_1801, out=buf1543)
        del permute_1801
        buf1546 = reinterpret_tensor(buf1543, (1, 128, 512), (65536, 512, 1), 0); del buf1543  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1546, le_76, 65536, grid=grid(65536), stream=stream0)
        del le_76
        buf1547 = buf1519; del buf1519  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1546, (128, 512), (512, 1), 0), permute_1805, out=buf1547)
        del permute_1805
        buf1540 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1541 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1545 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1550 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1551 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_74, attention_output_23, mul_39], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf1537, addmm_74, buf10, primals_77, primals_78, buf1542, buf1547, buf1540, buf1541, buf1545, buf1550, buf1551, 128, 128, grid=grid(128), stream=stream0)
        del addmm_74
        del primals_78
        buf1544 = buf1513; del buf1513  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1542, (128, 128), (1, 128), 0), view_198, out=buf1544)
        del view_198
        buf1548 = reinterpret_tensor(buf11, (512, 128), (128, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1546, (512, 128), (1, 512), 0), view_196, out=buf1548)
        del view_196
        buf1549 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1546, buf1549, 512, 128, grid=grid(512), stream=stream0)
        buf1552 = buf1542; del buf1542  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1552, buf1547, primals_77, 16384, grid=grid(16384), stream=stream0)
        del primals_77
        buf1553 = reinterpret_tensor(buf1546, (128, 512), (512, 1), 0); del buf1546  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1552, (128, 128), (128, 1), 0), permute_1809, out=buf1553)
        del permute_1809
        buf1554 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1552, (128, 128), (1, 128), 0), view_194, out=buf1554)
        del view_194
        buf1556 = reinterpret_tensor(buf1553, (1, 128, 512), (65536, 512, 1), 0); del buf1553  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1556, le_77, 65536, grid=grid(65536), stream=stream0)
        del le_77
        buf1557 = buf1547; del buf1547  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1556, (128, 512), (512, 1), 0), permute_1813, out=buf1557)
        del permute_1813
        buf1562 = reinterpret_tensor(buf1537, (1, 128, 128), (16384, 128, 1), 0); del buf1537  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf1552, buf1557, primals_75, buf1562, 16384, grid=grid(16384), stream=stream0)
        del primals_75
        buf1563 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1562, (128, 128), (128, 1), 0), permute_1817, out=buf1563)
        del permute_1817
        buf1566 = reinterpret_tensor(buf1563, (1, 128, 512), (65536, 512, 1), 0); del buf1563  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1566, le_78, 65536, grid=grid(65536), stream=stream0)
        del le_78
        buf1567 = reinterpret_tensor(buf10, (128, 128), (128, 1), 0); del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1566, (128, 512), (512, 1), 0), permute_1821, out=buf1567)
        del permute_1821
        buf1555 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1560 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1561 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1565 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1570 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1571 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_70, attention_output_21, mul_37], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf1552, buf1557, addmm_70, buf9, primals_73, primals_74, buf1562, buf1567, buf1555, buf1560, buf1561, buf1565, buf1570, buf1571, 128, 128, grid=grid(128), stream=stream0)
        del addmm_70
        del buf1552
        del primals_74
        buf1558 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1556, (512, 128), (1, 512), 0), view_192, out=buf1558)
        del view_192
        buf1559 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1556, buf1559, 512, 128, grid=grid(512), stream=stream0)
        buf1564 = reinterpret_tensor(buf1556, (128, 512), (512, 1), 0); del buf1556  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1562, (128, 128), (1, 128), 0), view_190, out=buf1564)
        del view_190
        buf1568 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1566, (512, 128), (1, 512), 0), view_188, out=buf1568)
        del view_188
        buf1569 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1566, buf1569, 512, 128, grid=grid(512), stream=stream0)
        buf1572 = buf1562; del buf1562  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1572, buf1567, primals_73, 16384, grid=grid(16384), stream=stream0)
        del primals_73
        buf1573 = reinterpret_tensor(buf1566, (128, 512), (512, 1), 0); del buf1566  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1572, (128, 128), (128, 1), 0), permute_1825, out=buf1573)
        del permute_1825
        buf1574 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1572, (128, 128), (1, 128), 0), view_186, out=buf1574)
        del view_186
        buf1576 = reinterpret_tensor(buf1573, (1, 128, 512), (65536, 512, 1), 0); del buf1573  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1576, le_79, 65536, grid=grid(65536), stream=stream0)
        del le_79
        buf1577 = buf1567; del buf1567  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1576, (128, 512), (512, 1), 0), permute_1829, out=buf1577)
        del permute_1829
        buf1582 = buf9; del buf9  # reuse
        buf1607 = reinterpret_tensor(buf1557, (1, 128, 128), (16384, 128, 1), 0); del buf1557  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf1572, buf1577, primals_71, primals_67, buf1582, buf1607, 16384, grid=grid(16384), stream=stream0)
        del primals_71
        buf1575 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1580 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1581 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1585 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1605 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1606 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_66, layer_input_24, mul_34], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf1572, buf1577, addmm_66, addmm_61, primals_67, primals_68, buf1582, buf1575, buf1580, buf1581, buf1585, buf1605, buf1606, 128, 128, grid=grid(128), stream=stream0)
        del addmm_61
        del addmm_66
        del primals_67
        del primals_68
        buf1578 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1576, (512, 128), (1, 512), 0), view_184, out=buf1578)
        del view_184
        buf1579 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1576, buf1579, 512, 128, grid=grid(512), stream=stream0)
        buf1583 = buf1577; del buf1577  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1582, (128, 128), (128, 1), 0), permute_1833, out=buf1583)
        del permute_1833
        buf1584 = reinterpret_tensor(buf1572, (128, 128), (128, 1), 0); del buf1572  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1582, (128, 128), (1, 128), 0), view_182, out=buf1584)
        del view_182
        # Source Nodes: [], Original ATen: []
        buf1586 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1583, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_57, clone_default_58, clone_default_59, None, alias_default_39, getitem_184, getitem_185, getitem_186, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_39
        del clone_default_57
        del clone_default_58
        del clone_default_59
        del getitem_184
        del getitem_185
        del getitem_186
        buf1587 = buf1586[0]
        buf1588 = buf1586[1]
        buf1589 = buf1586[2]
        del buf1586
        buf1590 = reinterpret_tensor(buf1576, (128, 512), (512, 1), 0); del buf1576  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1589, (128, 128), (128, 1), 0), permute_1846, out=buf1590)
        del permute_1846
        buf1591 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1589, (128, 128), (1, 128), 0), view_162, out=buf1591)
        buf1592 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1589, buf1592, 128, 128, grid=grid(128), stream=stream0)
        buf1593 = reinterpret_tensor(buf1589, (128, 128), (128, 1), 0); del buf1589  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1588, (128, 128), (128, 1), 0), permute_1850, out=buf1593)
        del permute_1850
        buf1594 = buf1583; del buf1583  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1588, (128, 128), (1, 128), 0), view_166, out=buf1594)
        buf1595 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1588, buf1595, 128, 128, grid=grid(128), stream=stream0)
        buf1596 = reinterpret_tensor(buf1588, (128, 128), (128, 1), 0); del buf1588  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1587, (128, 128), (128, 1), 0), permute_1854, out=buf1596)
        del permute_1854
        buf1597 = reinterpret_tensor(buf1582, (128, 128), (128, 1), 0); del buf1582  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1587, (128, 128), (1, 128), 0), view_166, out=buf1597)
        del view_166
        buf1598 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1587, buf1598, 128, 128, grid=grid(128), stream=stream0)
        del buf1587
        buf1599 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1600 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf1593, buf1596, addmm_62, buf1599, buf1600, 128, 128, grid=grid(128), stream=stream0)
        del addmm_62
        buf1601 = reinterpret_tensor(buf1593, (1, 128, 128), (16384, 128, 1), 0); del buf1593  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1601, buf1596, primals_69, 16384, grid=grid(16384), stream=stream0)
        del primals_69
        buf1602 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1601, (128, 128), (128, 1), 0), permute_1858, out=buf1602)
        del permute_1858
        buf1603 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1601, (128, 128), (1, 128), 0), view_162, out=buf1603)
        buf1604 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1601, buf1604, 128, 128, grid=grid(128), stream=stream0)
        buf1608 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1607, (128, 128), (128, 1), 0), permute_1862, out=buf1608)
        del permute_1862
        buf1609 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1607, (128, 128), (1, 128), 0), view_162, out=buf1609)
        del view_162
        buf1610 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1607, buf1610, 128, 128, grid=grid(128), stream=stream0)
        buf1611 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1613 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_61, mul_25, value_tensor_3], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_24.run(buf1535, buf1590, buf1602, buf1608, addmm_60, buf6, primals_49, primals_50, buf1611, buf1613, 512, 128, grid=grid(512), stream=stream0)
        del addmm_60
        del primals_50
        buf1612 = buf1535; del buf1535  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf1612, buf1590, buf1602, buf1608, primals_65, 65536, grid=grid(65536), stream=stream0)
        del primals_65
        buf1614 = reinterpret_tensor(buf1607, (128, 128), (128, 1), 0); del buf1607  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1612, (128, 512), (512, 1), 0), permute_1866, out=buf1614)
        del permute_1866
        buf1615 = reinterpret_tensor(buf1608, (512, 128), (128, 1), 0); del buf1608  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1612, (512, 128), (1, 512), 0), view_160, out=buf1615)
        del view_160
        buf1616 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf1612, buf1616, 512, 128, grid=grid(512), stream=stream0)
        buf1619 = buf1601; del buf1601  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf1614, primals_63, buf1619, 16384, grid=grid(16384), stream=stream0)
        del primals_63
        buf1620 = buf1602; del buf1602  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1619, (128, 128), (128, 1), 0), permute_1870, out=buf1620)
        del permute_1870
        buf1623 = reinterpret_tensor(buf1620, (1, 128, 512), (65536, 512, 1), 0); del buf1620  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1623, le_80, 65536, grid=grid(65536), stream=stream0)
        del le_80
        buf1624 = buf1596; del buf1596  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1623, (128, 512), (512, 1), 0), permute_1874, out=buf1624)
        del permute_1874
        buf1617 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1618 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1622 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1627 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1628 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_59, attention_output_18, mul_31], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf1614, addmm_59, buf8, primals_61, primals_62, buf1619, buf1624, buf1617, buf1618, buf1622, buf1627, buf1628, 128, 128, grid=grid(128), stream=stream0)
        del addmm_59
        del primals_62
        buf1621 = buf1590; del buf1590  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1619, (128, 128), (1, 128), 0), view_158, out=buf1621)
        del view_158
        buf1625 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1623, (512, 128), (1, 512), 0), view_156, out=buf1625)
        del view_156
        buf1626 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1623, buf1626, 512, 128, grid=grid(512), stream=stream0)
        buf1629 = buf1619; del buf1619  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1629, buf1624, primals_61, 16384, grid=grid(16384), stream=stream0)
        del primals_61
        buf1630 = reinterpret_tensor(buf1623, (128, 512), (512, 1), 0); del buf1623  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1629, (128, 128), (128, 1), 0), permute_1878, out=buf1630)
        del permute_1878
        buf1631 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1629, (128, 128), (1, 128), 0), view_154, out=buf1631)
        del view_154
        buf1633 = reinterpret_tensor(buf1630, (1, 128, 512), (65536, 512, 1), 0); del buf1630  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1633, le_81, 65536, grid=grid(65536), stream=stream0)
        del le_81
        buf1634 = buf1624; del buf1624  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1633, (128, 512), (512, 1), 0), permute_1882, out=buf1634)
        del permute_1882
        buf1639 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf1629, buf1634, primals_59, buf1639, 16384, grid=grid(16384), stream=stream0)
        del primals_59
        buf1640 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1639, (128, 128), (128, 1), 0), permute_1886, out=buf1640)
        del permute_1886
        buf1643 = reinterpret_tensor(buf1640, (1, 128, 512), (65536, 512, 1), 0); del buf1640  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1643, le_82, 65536, grid=grid(65536), stream=stream0)
        del le_82
        buf1644 = buf1614; del buf1614  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1643, (128, 512), (512, 1), 0), permute_1890, out=buf1644)
        del permute_1890
        buf1632 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1637 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1638 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1642 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1647 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1648 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_55, attention_output_16, mul_29], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf1629, buf1634, addmm_55, buf7, primals_57, primals_58, buf1639, buf1644, buf1632, buf1637, buf1638, buf1642, buf1647, buf1648, 128, 128, grid=grid(128), stream=stream0)
        del addmm_55
        del buf1629
        del primals_58
        buf1635 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1633, (512, 128), (1, 512), 0), view_152, out=buf1635)
        del view_152
        buf1636 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1633, buf1636, 512, 128, grid=grid(512), stream=stream0)
        buf1641 = reinterpret_tensor(buf1633, (128, 512), (512, 1), 0); del buf1633  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1639, (128, 128), (1, 128), 0), view_150, out=buf1641)
        del view_150
        buf1645 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1643, (512, 128), (1, 512), 0), view_148, out=buf1645)
        del view_148
        buf1646 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1643, buf1646, 512, 128, grid=grid(512), stream=stream0)
        buf1649 = buf1639; del buf1639  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1649, buf1644, primals_57, 16384, grid=grid(16384), stream=stream0)
        del primals_57
        buf1650 = reinterpret_tensor(buf1643, (128, 512), (512, 1), 0); del buf1643  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1649, (128, 128), (128, 1), 0), permute_1894, out=buf1650)
        del permute_1894
        buf1651 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1649, (128, 128), (1, 128), 0), view_146, out=buf1651)
        del view_146
        buf1653 = reinterpret_tensor(buf1650, (1, 128, 512), (65536, 512, 1), 0); del buf1650  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1653, le_83, 65536, grid=grid(65536), stream=stream0)
        del le_83
        buf1654 = buf1644; del buf1644  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1653, (128, 512), (512, 1), 0), permute_1898, out=buf1654)
        del permute_1898
        buf1659 = buf7; del buf7  # reuse
        buf1684 = reinterpret_tensor(buf1634, (1, 128, 128), (16384, 128, 1), 0); del buf1634  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf1649, buf1654, primals_55, primals_51, buf1659, buf1684, 16384, grid=grid(16384), stream=stream0)
        del primals_55
        buf1652 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1657 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1658 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1662 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1682 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1683 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_51, layer_input_19, mul_26], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf1649, buf1654, addmm_51, addmm_46, primals_51, primals_52, buf1659, buf1652, buf1657, buf1658, buf1662, buf1682, buf1683, 128, 128, grid=grid(128), stream=stream0)
        del addmm_46
        del addmm_51
        del primals_51
        del primals_52
        buf1655 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1653, (512, 128), (1, 512), 0), view_144, out=buf1655)
        del view_144
        buf1656 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1653, buf1656, 512, 128, grid=grid(512), stream=stream0)
        buf1660 = buf1654; del buf1654  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1659, (128, 128), (128, 1), 0), permute_1902, out=buf1660)
        del permute_1902
        buf1661 = reinterpret_tensor(buf1649, (128, 128), (128, 1), 0); del buf1649  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1659, (128, 128), (1, 128), 0), view_142, out=buf1661)
        del view_142
        # Source Nodes: [], Original ATen: []
        buf1663 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1660, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_60, clone_default_61, clone_default_62, None, alias_default_41, getitem_191, getitem_192, getitem_193, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_41
        del clone_default_60
        del clone_default_61
        del clone_default_62
        del getitem_191
        del getitem_192
        del getitem_193
        buf1664 = buf1663[0]
        buf1665 = buf1663[1]
        buf1666 = buf1663[2]
        del buf1663
        buf1667 = reinterpret_tensor(buf1653, (128, 512), (512, 1), 0); del buf1653  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1666, (128, 128), (128, 1), 0), permute_1915, out=buf1667)
        del permute_1915
        buf1668 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1666, (128, 128), (1, 128), 0), view_122, out=buf1668)
        buf1669 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1666, buf1669, 128, 128, grid=grid(128), stream=stream0)
        buf1670 = reinterpret_tensor(buf1666, (128, 128), (128, 1), 0); del buf1666  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1665, (128, 128), (128, 1), 0), permute_1919, out=buf1670)
        del permute_1919
        buf1671 = buf1660; del buf1660  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1665, (128, 128), (1, 128), 0), view_126, out=buf1671)
        buf1672 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1665, buf1672, 128, 128, grid=grid(128), stream=stream0)
        buf1673 = reinterpret_tensor(buf1665, (128, 128), (128, 1), 0); del buf1665  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1664, (128, 128), (128, 1), 0), permute_1923, out=buf1673)
        del permute_1923
        buf1674 = reinterpret_tensor(buf1659, (128, 128), (128, 1), 0); del buf1659  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1664, (128, 128), (1, 128), 0), view_126, out=buf1674)
        del view_126
        buf1675 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1664, buf1675, 128, 128, grid=grid(128), stream=stream0)
        del buf1664
        buf1676 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1677 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf1670, buf1673, addmm_47, buf1676, buf1677, 128, 128, grid=grid(128), stream=stream0)
        del addmm_47
        buf1678 = reinterpret_tensor(buf1670, (1, 128, 128), (16384, 128, 1), 0); del buf1670  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1678, buf1673, primals_53, 16384, grid=grid(16384), stream=stream0)
        del primals_53
        buf1679 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1678, (128, 128), (128, 1), 0), permute_1927, out=buf1679)
        del permute_1927
        buf1680 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1678, (128, 128), (1, 128), 0), view_122, out=buf1680)
        buf1681 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1678, buf1681, 128, 128, grid=grid(128), stream=stream0)
        buf1685 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1684, (128, 128), (128, 1), 0), permute_1931, out=buf1685)
        del permute_1931
        buf1686 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1684, (128, 128), (1, 128), 0), view_122, out=buf1686)
        del view_122
        buf1687 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1684, buf1687, 128, 128, grid=grid(128), stream=stream0)
        buf1688 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1690 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_22.run(buf1612, buf1667, buf1679, buf1685, buf6, buf1688, buf1690, 512, 128, grid=grid(512), stream=stream0)
        buf1689 = buf1612; del buf1612  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf1689, buf1667, buf1679, buf1685, primals_49, 65536, grid=grid(65536), stream=stream0)
        del primals_49
        buf1691 = reinterpret_tensor(buf1684, (128, 128), (128, 1), 0); del buf1684  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1689, (128, 512), (512, 1), 0), permute_1935, out=buf1691)
        del permute_1935
        buf1692 = reinterpret_tensor(buf1685, (512, 128), (128, 1), 0); del buf1685  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1689, (512, 128), (1, 512), 0), view_120, out=buf1692)
        del view_120
        buf1693 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf1689, buf1693, 512, 128, grid=grid(512), stream=stream0)
        buf1696 = buf1678; del buf1678  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf1691, primals_47, buf1696, 16384, grid=grid(16384), stream=stream0)
        del primals_47
        buf1697 = buf1679; del buf1679  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1696, (128, 128), (128, 1), 0), permute_1939, out=buf1697)
        del permute_1939
        buf1700 = reinterpret_tensor(buf1697, (1, 128, 512), (65536, 512, 1), 0); del buf1697  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1700, le_84, 65536, grid=grid(65536), stream=stream0)
        del le_84
        buf1701 = buf1673; del buf1673  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1700, (128, 512), (512, 1), 0), permute_1943, out=buf1701)
        del permute_1943
        buf1694 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1695 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1699 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1704 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1705 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_44, attention_output_13, mul_23], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf1691, addmm_44, buf5, primals_45, primals_46, buf1696, buf1701, buf1694, buf1695, buf1699, buf1704, buf1705, 128, 128, grid=grid(128), stream=stream0)
        del addmm_44
        del primals_46
        buf1698 = buf1667; del buf1667  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1696, (128, 128), (1, 128), 0), view_118, out=buf1698)
        del view_118
        buf1702 = reinterpret_tensor(buf6, (512, 128), (128, 1), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1700, (512, 128), (1, 512), 0), view_116, out=buf1702)
        del view_116
        buf1703 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1700, buf1703, 512, 128, grid=grid(512), stream=stream0)
        buf1706 = buf1696; del buf1696  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1706, buf1701, primals_45, 16384, grid=grid(16384), stream=stream0)
        del primals_45
        buf1707 = reinterpret_tensor(buf1700, (128, 512), (512, 1), 0); del buf1700  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1706, (128, 128), (128, 1), 0), permute_1947, out=buf1707)
        del permute_1947
        buf1708 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1706, (128, 128), (1, 128), 0), view_114, out=buf1708)
        del view_114
        buf1710 = reinterpret_tensor(buf1707, (1, 128, 512), (65536, 512, 1), 0); del buf1707  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1710, le_85, 65536, grid=grid(65536), stream=stream0)
        del le_85
        buf1711 = buf1701; del buf1701  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1710, (128, 512), (512, 1), 0), permute_1951, out=buf1711)
        del permute_1951
        buf1716 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf1706, buf1711, primals_43, buf1716, 16384, grid=grid(16384), stream=stream0)
        del primals_43
        buf1717 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1716, (128, 128), (128, 1), 0), permute_1955, out=buf1717)
        del permute_1955
        buf1720 = reinterpret_tensor(buf1717, (1, 128, 512), (65536, 512, 1), 0); del buf1717  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1720, le_86, 65536, grid=grid(65536), stream=stream0)
        del le_86
        buf1721 = buf1691; del buf1691  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1720, (128, 512), (512, 1), 0), permute_1959, out=buf1721)
        del permute_1959
        buf1709 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1714 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1715 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1719 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1724 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1725 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_40, attention_output_11, mul_21], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf1706, buf1711, addmm_40, buf4, primals_41, primals_42, buf1716, buf1721, buf1709, buf1714, buf1715, buf1719, buf1724, buf1725, 128, 128, grid=grid(128), stream=stream0)
        del addmm_40
        del buf1706
        del primals_42
        buf1712 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1710, (512, 128), (1, 512), 0), view_112, out=buf1712)
        del view_112
        buf1713 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1710, buf1713, 512, 128, grid=grid(512), stream=stream0)
        buf1718 = reinterpret_tensor(buf1710, (128, 512), (512, 1), 0); del buf1710  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1716, (128, 128), (1, 128), 0), view_110, out=buf1718)
        del view_110
        buf1722 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1720, (512, 128), (1, 512), 0), view_108, out=buf1722)
        del view_108
        buf1723 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1720, buf1723, 512, 128, grid=grid(512), stream=stream0)
        buf1726 = buf1716; del buf1716  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1726, buf1721, primals_41, 16384, grid=grid(16384), stream=stream0)
        del primals_41
        buf1727 = reinterpret_tensor(buf1720, (128, 512), (512, 1), 0); del buf1720  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1726, (128, 128), (128, 1), 0), permute_1963, out=buf1727)
        del permute_1963
        buf1728 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1726, (128, 128), (1, 128), 0), view_106, out=buf1728)
        del view_106
        buf1730 = reinterpret_tensor(buf1727, (1, 128, 512), (65536, 512, 1), 0); del buf1727  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1730, le_87, 65536, grid=grid(65536), stream=stream0)
        del le_87
        buf1731 = buf1721; del buf1721  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1730, (128, 512), (512, 1), 0), permute_1967, out=buf1731)
        del permute_1967
        buf1736 = buf4; del buf4  # reuse
        buf1761 = reinterpret_tensor(buf1711, (1, 128, 128), (16384, 128, 1), 0); del buf1711  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf1726, buf1731, primals_39, primals_35, buf1736, buf1761, 16384, grid=grid(16384), stream=stream0)
        del primals_39
        buf1729 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1734 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1735 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1739 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1759 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1760 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_36, layer_input_14, mul_18], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf1726, buf1731, addmm_36, addmm_31, primals_35, primals_36, buf1736, buf1729, buf1734, buf1735, buf1739, buf1759, buf1760, 128, 128, grid=grid(128), stream=stream0)
        del addmm_31
        del addmm_36
        del primals_35
        del primals_36
        buf1732 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1730, (512, 128), (1, 512), 0), view_104, out=buf1732)
        del view_104
        buf1733 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1730, buf1733, 512, 128, grid=grid(512), stream=stream0)
        buf1737 = buf1731; del buf1731  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1736, (128, 128), (128, 1), 0), permute_1971, out=buf1737)
        del permute_1971
        buf1738 = reinterpret_tensor(buf1726, (128, 128), (128, 1), 0); del buf1726  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1736, (128, 128), (1, 128), 0), view_102, out=buf1738)
        del view_102
        # Source Nodes: [], Original ATen: []
        buf1740 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1737, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_63, clone_default_64, clone_default_65, None, alias_default_43, getitem_198, getitem_199, getitem_200, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_43
        del clone_default_63
        del clone_default_64
        del clone_default_65
        del getitem_198
        del getitem_199
        del getitem_200
        buf1741 = buf1740[0]
        buf1742 = buf1740[1]
        buf1743 = buf1740[2]
        del buf1740
        buf1744 = reinterpret_tensor(buf1730, (128, 512), (512, 1), 0); del buf1730  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1743, (128, 128), (128, 1), 0), permute_1984, out=buf1744)
        del permute_1984
        buf1745 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1743, (128, 128), (1, 128), 0), view_82, out=buf1745)
        buf1746 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1743, buf1746, 128, 128, grid=grid(128), stream=stream0)
        buf1747 = reinterpret_tensor(buf1743, (128, 128), (128, 1), 0); del buf1743  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1742, (128, 128), (128, 1), 0), permute_1988, out=buf1747)
        del permute_1988
        buf1748 = buf1737; del buf1737  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1742, (128, 128), (1, 128), 0), view_86, out=buf1748)
        buf1749 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1742, buf1749, 128, 128, grid=grid(128), stream=stream0)
        buf1750 = reinterpret_tensor(buf1742, (128, 128), (128, 1), 0); del buf1742  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1741, (128, 128), (128, 1), 0), permute_1992, out=buf1750)
        del permute_1992
        buf1751 = reinterpret_tensor(buf1736, (128, 128), (128, 1), 0); del buf1736  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1741, (128, 128), (1, 128), 0), view_86, out=buf1751)
        del view_86
        buf1752 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1741, buf1752, 128, 128, grid=grid(128), stream=stream0)
        del buf1741
        buf1753 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1754 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf1747, buf1750, addmm_32, buf1753, buf1754, 128, 128, grid=grid(128), stream=stream0)
        del addmm_32
        buf1755 = reinterpret_tensor(buf1747, (1, 128, 128), (16384, 128, 1), 0); del buf1747  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1755, buf1750, primals_37, 16384, grid=grid(16384), stream=stream0)
        del primals_37
        buf1756 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1755, (128, 128), (128, 1), 0), permute_1996, out=buf1756)
        del permute_1996
        buf1757 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1755, (128, 128), (1, 128), 0), view_82, out=buf1757)
        buf1758 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1755, buf1758, 128, 128, grid=grid(128), stream=stream0)
        buf1762 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1761, (128, 128), (128, 1), 0), permute_2000, out=buf1762)
        del permute_2000
        buf1763 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1761, (128, 128), (1, 128), 0), view_82, out=buf1763)
        del view_82
        buf1764 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1761, buf1764, 128, 128, grid=grid(128), stream=stream0)
        buf1765 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1767 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_31, mul_9, value_tensor_1], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_24.run(buf1689, buf1744, buf1756, buf1762, addmm_30, add_16, primals_17, primals_18, buf1765, buf1767, 512, 128, grid=grid(512), stream=stream0)
        del addmm_30
        del primals_18
        buf1766 = buf1689; del buf1689  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf1766, buf1744, buf1756, buf1762, primals_33, 65536, grid=grid(65536), stream=stream0)
        del primals_33
        buf1768 = reinterpret_tensor(buf1761, (128, 128), (128, 1), 0); del buf1761  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1766, (128, 512), (512, 1), 0), permute_2004, out=buf1768)
        del permute_2004
        buf1769 = reinterpret_tensor(buf1762, (512, 128), (128, 1), 0); del buf1762  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1766, (512, 128), (1, 512), 0), view_80, out=buf1769)
        del view_80
        buf1770 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf1766, buf1770, 512, 128, grid=grid(512), stream=stream0)
        buf1773 = buf1755; del buf1755  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf1768, primals_31, buf1773, 16384, grid=grid(16384), stream=stream0)
        del primals_31
        buf1774 = buf1756; del buf1756  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1773, (128, 128), (128, 1), 0), permute_2008, out=buf1774)
        del permute_2008
        buf1777 = reinterpret_tensor(buf1774, (1, 128, 512), (65536, 512, 1), 0); del buf1774  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1777, le_88, 65536, grid=grid(65536), stream=stream0)
        del le_88
        buf1778 = buf1750; del buf1750  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1777, (128, 512), (512, 1), 0), permute_2012, out=buf1778)
        del permute_2012
        buf1771 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1772 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1776 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1781 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1782 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_29, attention_output_8, mul_15], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf1768, addmm_29, buf3, primals_29, primals_30, buf1773, buf1778, buf1771, buf1772, buf1776, buf1781, buf1782, 128, 128, grid=grid(128), stream=stream0)
        del addmm_29
        del primals_30
        buf1775 = buf1744; del buf1744  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1773, (128, 128), (1, 128), 0), view_78, out=buf1775)
        del view_78
        buf1779 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1777, (512, 128), (1, 512), 0), view_76, out=buf1779)
        del view_76
        buf1780 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1777, buf1780, 512, 128, grid=grid(512), stream=stream0)
        buf1783 = buf1773; del buf1773  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1783, buf1778, primals_29, 16384, grid=grid(16384), stream=stream0)
        del primals_29
        buf1784 = reinterpret_tensor(buf1777, (128, 512), (512, 1), 0); del buf1777  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1783, (128, 128), (128, 1), 0), permute_2016, out=buf1784)
        del permute_2016
        buf1785 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1783, (128, 128), (1, 128), 0), view_74, out=buf1785)
        del view_74
        buf1787 = reinterpret_tensor(buf1784, (1, 128, 512), (65536, 512, 1), 0); del buf1784  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1787, le_89, 65536, grid=grid(65536), stream=stream0)
        del le_89
        buf1788 = buf1778; del buf1778  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1787, (128, 512), (512, 1), 0), permute_2020, out=buf1788)
        del permute_2020
        buf1793 = buf3; del buf3  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf1783, buf1788, primals_27, buf1793, 16384, grid=grid(16384), stream=stream0)
        del primals_27
        buf1794 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1793, (128, 128), (128, 1), 0), permute_2024, out=buf1794)
        del permute_2024
        buf1797 = reinterpret_tensor(buf1794, (1, 128, 512), (65536, 512, 1), 0); del buf1794  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1797, le_90, 65536, grid=grid(65536), stream=stream0)
        del le_90
        buf1798 = buf1768; del buf1768  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1797, (128, 512), (512, 1), 0), permute_2028, out=buf1798)
        del permute_2028
        buf1786 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1791 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1792 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1796 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1801 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1802 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_25, attention_output_6, mul_13], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf1783, buf1788, addmm_25, buf2, primals_25, primals_26, buf1793, buf1798, buf1786, buf1791, buf1792, buf1796, buf1801, buf1802, 128, 128, grid=grid(128), stream=stream0)
        del addmm_25
        del buf1783
        del primals_26
        buf1789 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1787, (512, 128), (1, 512), 0), view_72, out=buf1789)
        del view_72
        buf1790 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1787, buf1790, 512, 128, grid=grid(512), stream=stream0)
        buf1795 = reinterpret_tensor(buf1787, (128, 512), (512, 1), 0); del buf1787  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1793, (128, 128), (1, 128), 0), view_70, out=buf1795)
        del view_70
        buf1799 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1797, (512, 128), (1, 512), 0), view_68, out=buf1799)
        del view_68
        buf1800 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1797, buf1800, 512, 128, grid=grid(512), stream=stream0)
        buf1803 = buf1793; del buf1793  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1803, buf1798, primals_25, 16384, grid=grid(16384), stream=stream0)
        del primals_25
        buf1804 = reinterpret_tensor(buf1797, (128, 512), (512, 1), 0); del buf1797  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1803, (128, 128), (128, 1), 0), permute_2032, out=buf1804)
        del permute_2032
        buf1805 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1803, (128, 128), (1, 128), 0), view_66, out=buf1805)
        del view_66
        buf1807 = reinterpret_tensor(buf1804, (1, 128, 512), (65536, 512, 1), 0); del buf1804  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1807, le_91, 65536, grid=grid(65536), stream=stream0)
        del le_91
        buf1808 = buf1798; del buf1798  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1807, (128, 512), (512, 1), 0), permute_2036, out=buf1808)
        del permute_2036
        buf1813 = buf2; del buf2  # reuse
        buf1838 = reinterpret_tensor(buf1788, (1, 128, 128), (16384, 128, 1), 0); del buf1788  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf1803, buf1808, primals_23, primals_19, buf1813, buf1838, 16384, grid=grid(16384), stream=stream0)
        del primals_23
        buf1806 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1811 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1812 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1816 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1836 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1837 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_21, layer_input_9, mul_10], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf1803, buf1808, addmm_21, addmm_16, primals_19, primals_20, buf1813, buf1806, buf1811, buf1812, buf1816, buf1836, buf1837, 128, 128, grid=grid(128), stream=stream0)
        del addmm_16
        del addmm_21
        del primals_19
        del primals_20
        buf1809 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1807, (512, 128), (1, 512), 0), view_64, out=buf1809)
        del view_64
        buf1810 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1807, buf1810, 512, 128, grid=grid(512), stream=stream0)
        buf1814 = buf1808; del buf1808  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1813, (128, 128), (128, 1), 0), permute_2040, out=buf1814)
        del permute_2040
        buf1815 = reinterpret_tensor(buf1803, (128, 128), (128, 1), 0); del buf1803  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1813, (128, 128), (1, 128), 0), view_62, out=buf1815)
        del view_62
        # Source Nodes: [], Original ATen: []
        buf1817 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1814, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_66, clone_default_67, clone_default_68, None, alias_default_45, getitem_205, getitem_206, getitem_207, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_45
        del clone_default_66
        del clone_default_67
        del clone_default_68
        del getitem_205
        del getitem_206
        del getitem_207
        buf1818 = buf1817[0]
        buf1819 = buf1817[1]
        buf1820 = buf1817[2]
        del buf1817
        buf1821 = reinterpret_tensor(buf1807, (128, 512), (512, 1), 0); del buf1807  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1820, (128, 128), (128, 1), 0), permute_2053, out=buf1821)
        del permute_2053
        buf1822 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1820, (128, 128), (1, 128), 0), view_42, out=buf1822)
        buf1823 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1820, buf1823, 128, 128, grid=grid(128), stream=stream0)
        buf1824 = reinterpret_tensor(buf1820, (128, 128), (128, 1), 0); del buf1820  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1819, (128, 128), (128, 1), 0), permute_2057, out=buf1824)
        del permute_2057
        buf1825 = buf1814; del buf1814  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1819, (128, 128), (1, 128), 0), view_46, out=buf1825)
        buf1826 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1819, buf1826, 128, 128, grid=grid(128), stream=stream0)
        buf1827 = reinterpret_tensor(buf1819, (128, 128), (128, 1), 0); del buf1819  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1818, (128, 128), (128, 1), 0), permute_2061, out=buf1827)
        del permute_2061
        buf1828 = reinterpret_tensor(buf1813, (128, 128), (128, 1), 0); del buf1813  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1818, (128, 128), (1, 128), 0), view_46, out=buf1828)
        del view_46
        buf1829 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1818, buf1829, 128, 128, grid=grid(128), stream=stream0)
        del buf1818
        buf1830 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1831 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf1824, buf1827, addmm_17, buf1830, buf1831, 128, 128, grid=grid(128), stream=stream0)
        del addmm_17
        buf1832 = reinterpret_tensor(buf1824, (1, 128, 128), (16384, 128, 1), 0); del buf1824  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1832, buf1827, primals_21, 16384, grid=grid(16384), stream=stream0)
        del primals_21
        buf1833 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1832, (128, 128), (128, 1), 0), permute_2065, out=buf1833)
        del permute_2065
        buf1834 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1832, (128, 128), (1, 128), 0), view_42, out=buf1834)
        buf1835 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1832, buf1835, 128, 128, grid=grid(128), stream=stream0)
        buf1839 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1838, (128, 128), (128, 1), 0), permute_2069, out=buf1839)
        del permute_2069
        buf1840 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1838, (128, 128), (1, 128), 0), view_42, out=buf1840)
        del view_42
        buf1841 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1838, buf1841, 128, 128, grid=grid(128), stream=stream0)
        buf1842 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1844 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_22.run(buf1766, buf1821, buf1833, buf1839, add_16, buf1842, buf1844, 512, 128, grid=grid(512), stream=stream0)
        del add_16
        buf1843 = buf1766; del buf1766  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_25.run(buf1843, buf1821, buf1833, buf1839, primals_17, 65536, grid=grid(65536), stream=stream0)
        del primals_17
        buf1845 = reinterpret_tensor(buf1838, (128, 128), (128, 1), 0); del buf1838  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1843, (128, 512), (512, 1), 0), permute_2073, out=buf1845)
        del permute_2073
        buf1846 = reinterpret_tensor(buf1839, (512, 128), (128, 1), 0); del buf1839  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1843, (512, 128), (1, 512), 0), view_40, out=buf1846)
        del view_40
        buf1847 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf1843, buf1847, 512, 128, grid=grid(512), stream=stream0)
        buf1850 = buf1832; del buf1832  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf1845, primals_15, buf1850, 16384, grid=grid(16384), stream=stream0)
        del primals_15
        buf1851 = buf1833; del buf1833  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1850, (128, 128), (128, 1), 0), permute_2077, out=buf1851)
        del permute_2077
        buf1854 = reinterpret_tensor(buf1851, (1, 128, 512), (65536, 512, 1), 0); del buf1851  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1854, le_92, 65536, grid=grid(65536), stream=stream0)
        del le_92
        buf1855 = buf1827; del buf1827  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1854, (128, 512), (512, 1), 0), permute_2081, out=buf1855)
        del permute_2081
        buf1848 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1849 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1853 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1858 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1859 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_14, attention_output_3, mul_7], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_13.run(buf1845, addmm_14, buf1, primals_13, primals_14, buf1850, buf1855, buf1848, buf1849, buf1853, buf1858, buf1859, 128, 128, grid=grid(128), stream=stream0)
        del addmm_14
        del primals_14
        buf1852 = buf1821; del buf1821  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1850, (128, 128), (1, 128), 0), view_38, out=buf1852)
        del view_38
        buf1856 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1854, (512, 128), (1, 512), 0), view_36, out=buf1856)
        del view_36
        buf1857 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1854, buf1857, 512, 128, grid=grid(512), stream=stream0)
        buf1860 = buf1850; del buf1850  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1860, buf1855, primals_13, 16384, grid=grid(16384), stream=stream0)
        del primals_13
        buf1861 = reinterpret_tensor(buf1854, (128, 512), (512, 1), 0); del buf1854  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1860, (128, 128), (128, 1), 0), permute_2085, out=buf1861)
        del permute_2085
        buf1862 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1860, (128, 128), (1, 128), 0), view_34, out=buf1862)
        del view_34
        buf1864 = reinterpret_tensor(buf1861, (1, 128, 512), (65536, 512, 1), 0); del buf1861  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1864, le_93, 65536, grid=grid(65536), stream=stream0)
        del le_93
        buf1865 = buf1855; del buf1855  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1864, (128, 512), (512, 1), 0), permute_2089, out=buf1865)
        del permute_2089
        buf1870 = reinterpret_tensor(buf1845, (1, 128, 128), (16384, 128, 1), 0); del buf1845  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_8.run(buf1860, buf1865, primals_11, buf1870, 16384, grid=grid(16384), stream=stream0)
        del primals_11
        buf1871 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1870, (128, 128), (128, 1), 0), permute_2093, out=buf1871)
        del permute_2093
        buf1874 = reinterpret_tensor(buf1871, (1, 128, 512), (65536, 512, 1), 0); del buf1871  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1874, le_94, 65536, grid=grid(65536), stream=stream0)
        del le_94
        buf1875 = reinterpret_tensor(buf1, (128, 128), (128, 1), 0); del buf1  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1874, (128, 512), (512, 1), 0), permute_2097, out=buf1875)
        del permute_2097
        buf1863 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1868 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1869 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1873 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1878 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1879 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_10, attention_output_1, mul_5], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_15.run(buf1860, buf1865, addmm_10, buf0, primals_9, primals_10, buf1870, buf1875, buf1863, buf1868, buf1869, buf1873, buf1878, buf1879, 128, 128, grid=grid(128), stream=stream0)
        del addmm_10
        del buf0
        del primals_10
        buf1866 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1864, (512, 128), (1, 512), 0), view_32, out=buf1866)
        del view_32
        buf1867 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1864, buf1867, 512, 128, grid=grid(512), stream=stream0)
        buf1872 = reinterpret_tensor(buf1864, (128, 512), (512, 1), 0); del buf1864  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1870, (128, 128), (1, 128), 0), view_30, out=buf1872)
        del view_30
        buf1876 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1874, (512, 128), (1, 512), 0), view_28, out=buf1876)
        del view_28
        buf1877 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1874, buf1877, 512, 128, grid=grid(512), stream=stream0)
        buf1880 = buf1870; del buf1870  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1880, buf1875, primals_9, 16384, grid=grid(16384), stream=stream0)
        del primals_9
        buf1881 = reinterpret_tensor(buf1874, (128, 512), (512, 1), 0); del buf1874  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1880, (128, 128), (128, 1), 0), permute_2101, out=buf1881)
        del permute_2101
        buf1882 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1880, (128, 128), (1, 128), 0), view_26, out=buf1882)
        del view_26
        buf1884 = reinterpret_tensor(buf1881, (1, 128, 512), (65536, 512, 1), 0); del buf1881  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
        triton_poi_fused_nll_loss_forward_threshold_backward_7.run(buf1884, le_95, 65536, grid=grid(65536), stream=stream0)
        del le_95
        buf1885 = buf1875; del buf1875  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1884, (128, 512), (512, 1), 0), permute_2105, out=buf1885)
        del permute_2105
        buf1890 = reinterpret_tensor(buf1865, (1, 128, 128), (16384, 128, 1), 0); del buf1865  # reuse
        buf1915 = buf1860; del buf1860  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_17.run(buf1880, buf1885, primals_7, primals_3, buf1890, buf1915, 16384, grid=grid(16384), stream=stream0)
        del primals_7
        buf1883 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1888 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1889 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1893 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf1913 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1914 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_6, layer_input_4, mul_2], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_18.run(buf1880, buf1885, addmm_6, addmm_1, primals_3, primals_4, buf1890, buf1883, buf1888, buf1889, buf1893, buf1913, buf1914, 128, 128, grid=grid(128), stream=stream0)
        del addmm_1
        del addmm_6
        del primals_3
        del primals_4
        buf1886 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1884, (512, 128), (1, 512), 0), view_24, out=buf1886)
        del view_24
        buf1887 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1884, buf1887, 512, 128, grid=grid(512), stream=stream0)
        buf1891 = buf1885; del buf1885  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1890, (128, 128), (128, 1), 0), permute_2109, out=buf1891)
        del permute_2109
        buf1892 = reinterpret_tensor(buf1880, (128, 128), (128, 1), 0); del buf1880  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1890, (128, 128), (1, 128), 0), view_22, out=buf1892)
        del view_22
        # Source Nodes: [], Original ATen: []
        buf1894 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1891, (1, 4, 128, 32), (16384, 32, 128, 1), 0), clone_default_69, clone_default_70, clone_default_71, None, alias_default_47, getitem_212, getitem_213, getitem_214, 0.1, [True, True, True, False], scale=0.17677669529663687)
        del alias_default_47
        del clone_default_69
        del clone_default_70
        del clone_default_71
        del getitem_212
        del getitem_213
        del getitem_214
        buf1895 = buf1894[0]
        buf1896 = buf1894[1]
        buf1897 = buf1894[2]
        del buf1894
        buf1898 = reinterpret_tensor(buf1884, (128, 512), (512, 1), 0); del buf1884  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1897, (128, 128), (128, 1), 0), permute_2122, out=buf1898)
        del permute_2122
        buf1899 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1897, (128, 128), (1, 128), 0), view_2, out=buf1899)
        buf1900 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1897, buf1900, 128, 128, grid=grid(128), stream=stream0)
        buf1901 = reinterpret_tensor(buf1897, (128, 128), (128, 1), 0); del buf1897  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1896, (128, 128), (128, 1), 0), permute_2126, out=buf1901)
        del permute_2126
        buf1902 = buf1891; del buf1891  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1896, (128, 128), (1, 128), 0), view_6, out=buf1902)
        buf1903 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1896, buf1903, 128, 128, grid=grid(128), stream=stream0)
        buf1904 = reinterpret_tensor(buf1896, (128, 128), (128, 1), 0); del buf1896  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1895, (128, 128), (128, 1), 0), permute_2130, out=buf1904)
        del permute_2130
        buf1905 = reinterpret_tensor(buf1890, (128, 128), (128, 1), 0); del buf1890  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1895, (128, 128), (1, 128), 0), view_6, out=buf1905)
        del view_6
        buf1906 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1895, buf1906, 128, 128, grid=grid(128), stream=stream0)
        del buf1895
        buf1907 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        buf1908 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_20.run(buf1901, buf1904, addmm_2, buf1907, buf1908, 128, 128, grid=grid(128), stream=stream0)
        del addmm_2
        buf1909 = reinterpret_tensor(buf1901, (1, 128, 128), (16384, 128, 1), 0); del buf1901  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_21.run(buf1909, buf1904, primals_5, 16384, grid=grid(16384), stream=stream0)
        del buf1904
        del primals_5
        buf1910 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1909, (128, 128), (128, 1), 0), permute_2134, out=buf1910)
        del permute_2134
        buf1911 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1909, (128, 128), (1, 128), 0), view_2, out=buf1911)
        buf1912 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1909, buf1912, 128, 128, grid=grid(128), stream=stream0)
        del buf1909
        buf1916 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1915, (128, 128), (128, 1), 0), permute_2138, out=buf1916)
        del permute_2138
        buf1917 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1915, (128, 128), (1, 128), 0), view_2, out=buf1917)
        del view_2
        buf1918 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf1915, buf1918, 128, 128, grid=grid(128), stream=stream0)
        buf1919 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        buf1921 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_add_mul_sum_22.run(buf1843, buf1898, buf1910, buf1916, add_1, buf1919, buf1921, 512, 128, grid=grid(512), stream=stream0)
        del add_1
        buf1920 = buf1843; del buf1843  # reuse
        buf1923 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        buf1927 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [start_loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.mul, aten.nll_loss_forward]
        triton_poi_fused_add_embedding_dense_backward_mul_nll_loss_forward_26.run(buf1920, buf1898, buf1910, buf1916, primals_1, slice_4, buf1923, buf1927, 65536, grid=grid(65536), stream=stream0)
        del buf1898
        del buf1910
        del buf1916
        del primals_1
        buf1922 = empty((2, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_27.run(buf1922, 1024, grid=grid(1024), stream=stream0)
        aten.index_put_(buf1922, [full_default], buf1923, True)
        del buf1923
        del full_default
        buf1926 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_28.run(buf1926, 262144, grid=grid(262144), stream=stream0)
        aten.index_put_(buf1926, [slice_4], buf1927, True)
        del buf1927
        del slice_4
        buf1930 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1920, (128, 512), (512, 1), 0), permute_2142, out=buf1930)
        del permute_2142
        buf1931 = empty((512, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1920, (512, 128), (1, 512), 0), view, out=buf1931)
        del view
        buf1932 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14.run(buf1920, buf1932, 512, 128, grid=grid(512), stream=stream0)
        del buf1920
        buf1933 = empty((30522, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_29.run(buf1933, 3906816, grid=grid(3906816), stream=stream0)
        buf1934 = buf1915; del buf1915  # reuse
        # Source Nodes: [start_loss], Original ATen: [aten.add, aten.constant_pad_nd, aten.embedding_dense_backward, aten.nll_loss_forward, aten.slice_backward]
        triton_poi_fused_add_constant_pad_nd_embedding_dense_backward_nll_loss_forward_slice_backward_30.run(primals_1115, buf1930, buf1934, 16384, grid=grid(16384), stream=stream0)
        del buf1930
        aten.index_put_(buf1933, [primals_1115], buf1934, True)
        del buf1934
        del primals_1115
        return (reinterpret_tensor(buf1921, (512, ), (1, ), 0), reinterpret_tensor(buf1919, (512, ), (1, ), 0), reinterpret_tensor(buf1914, (128, ), (1, ), 0), reinterpret_tensor(buf1913, (128, ), (1, ), 0), reinterpret_tensor(buf1908, (128, ), (1, ), 0), reinterpret_tensor(buf1907, (128, ), (1, ), 0), reinterpret_tensor(buf1889, (128, ), (1, ), 0), reinterpret_tensor(buf1888, (128, ), (1, ), 0), reinterpret_tensor(buf1879, (128, ), (1, ), 0), reinterpret_tensor(buf1878, (128, ), (1, ), 0), reinterpret_tensor(buf1869, (128, ), (1, ), 0), reinterpret_tensor(buf1868, (128, ), (1, ), 0), reinterpret_tensor(buf1859, (128, ), (1, ), 0), reinterpret_tensor(buf1858, (128, ), (1, ), 0), reinterpret_tensor(buf1849, (128, ), (1, ), 0), reinterpret_tensor(buf1848, (128, ), (1, ), 0), reinterpret_tensor(buf1844, (512, ), (1, ), 0), reinterpret_tensor(buf1842, (512, ), (1, ), 0), reinterpret_tensor(buf1837, (128, ), (1, ), 0), reinterpret_tensor(buf1836, (128, ), (1, ), 0), reinterpret_tensor(buf1831, (128, ), (1, ), 0), reinterpret_tensor(buf1830, (128, ), (1, ), 0), reinterpret_tensor(buf1812, (128, ), (1, ), 0), reinterpret_tensor(buf1811, (128, ), (1, ), 0), reinterpret_tensor(buf1802, (128, ), (1, ), 0), reinterpret_tensor(buf1801, (128, ), (1, ), 0), reinterpret_tensor(buf1792, (128, ), (1, ), 0), reinterpret_tensor(buf1791, (128, ), (1, ), 0), reinterpret_tensor(buf1782, (128, ), (1, ), 0), reinterpret_tensor(buf1781, (128, ), (1, ), 0), reinterpret_tensor(buf1772, (128, ), (1, ), 0), reinterpret_tensor(buf1771, (128, ), (1, ), 0), reinterpret_tensor(buf1767, (512, ), (1, ), 0), reinterpret_tensor(buf1765, (512, ), (1, ), 0), reinterpret_tensor(buf1760, (128, ), (1, ), 0), reinterpret_tensor(buf1759, (128, ), (1, ), 0), reinterpret_tensor(buf1754, (128, ), (1, ), 0), reinterpret_tensor(buf1753, (128, ), (1, ), 0), reinterpret_tensor(buf1735, (128, ), (1, ), 0), reinterpret_tensor(buf1734, (128, ), (1, ), 0), reinterpret_tensor(buf1725, (128, ), (1, ), 0), reinterpret_tensor(buf1724, (128, ), (1, ), 0), reinterpret_tensor(buf1715, (128, ), (1, ), 0), reinterpret_tensor(buf1714, (128, ), (1, ), 0), reinterpret_tensor(buf1705, (128, ), (1, ), 0), reinterpret_tensor(buf1704, (128, ), (1, ), 0), reinterpret_tensor(buf1695, (128, ), (1, ), 0), reinterpret_tensor(buf1694, (128, ), (1, ), 0), reinterpret_tensor(buf1690, (512, ), (1, ), 0), reinterpret_tensor(buf1688, (512, ), (1, ), 0), reinterpret_tensor(buf1683, (128, ), (1, ), 0), reinterpret_tensor(buf1682, (128, ), (1, ), 0), reinterpret_tensor(buf1677, (128, ), (1, ), 0), reinterpret_tensor(buf1676, (128, ), (1, ), 0), reinterpret_tensor(buf1658, (128, ), (1, ), 0), reinterpret_tensor(buf1657, (128, ), (1, ), 0), reinterpret_tensor(buf1648, (128, ), (1, ), 0), reinterpret_tensor(buf1647, (128, ), (1, ), 0), reinterpret_tensor(buf1638, (128, ), (1, ), 0), reinterpret_tensor(buf1637, (128, ), (1, ), 0), reinterpret_tensor(buf1628, (128, ), (1, ), 0), reinterpret_tensor(buf1627, (128, ), (1, ), 0), reinterpret_tensor(buf1618, (128, ), (1, ), 0), reinterpret_tensor(buf1617, (128, ), (1, ), 0), reinterpret_tensor(buf1613, (512, ), (1, ), 0), reinterpret_tensor(buf1611, (512, ), (1, ), 0), reinterpret_tensor(buf1606, (128, ), (1, ), 0), reinterpret_tensor(buf1605, (128, ), (1, ), 0), reinterpret_tensor(buf1600, (128, ), (1, ), 0), reinterpret_tensor(buf1599, (128, ), (1, ), 0), reinterpret_tensor(buf1581, (128, ), (1, ), 0), reinterpret_tensor(buf1580, (128, ), (1, ), 0), reinterpret_tensor(buf1571, (128, ), (1, ), 0), reinterpret_tensor(buf1570, (128, ), (1, ), 0), reinterpret_tensor(buf1561, (128, ), (1, ), 0), reinterpret_tensor(buf1560, (128, ), (1, ), 0), reinterpret_tensor(buf1551, (128, ), (1, ), 0), reinterpret_tensor(buf1550, (128, ), (1, ), 0), reinterpret_tensor(buf1541, (128, ), (1, ), 0), reinterpret_tensor(buf1540, (128, ), (1, ), 0), reinterpret_tensor(buf1536, (512, ), (1, ), 0), reinterpret_tensor(buf1534, (512, ), (1, ), 0), reinterpret_tensor(buf1529, (128, ), (1, ), 0), reinterpret_tensor(buf1528, (128, ), (1, ), 0), reinterpret_tensor(buf1523, (128, ), (1, ), 0), reinterpret_tensor(buf1522, (128, ), (1, ), 0), reinterpret_tensor(buf1504, (128, ), (1, ), 0), reinterpret_tensor(buf1503, (128, ), (1, ), 0), reinterpret_tensor(buf1494, (128, ), (1, ), 0), reinterpret_tensor(buf1493, (128, ), (1, ), 0), reinterpret_tensor(buf1484, (128, ), (1, ), 0), reinterpret_tensor(buf1483, (128, ), (1, ), 0), reinterpret_tensor(buf1474, (128, ), (1, ), 0), reinterpret_tensor(buf1473, (128, ), (1, ), 0), reinterpret_tensor(buf1464, (128, ), (1, ), 0), reinterpret_tensor(buf1463, (128, ), (1, ), 0), reinterpret_tensor(buf1459, (512, ), (1, ), 0), reinterpret_tensor(buf1457, (512, ), (1, ), 0), reinterpret_tensor(buf1452, (128, ), (1, ), 0), reinterpret_tensor(buf1451, (128, ), (1, ), 0), reinterpret_tensor(buf1446, (128, ), (1, ), 0), reinterpret_tensor(buf1445, (128, ), (1, ), 0), reinterpret_tensor(buf1427, (128, ), (1, ), 0), reinterpret_tensor(buf1426, (128, ), (1, ), 0), reinterpret_tensor(buf1417, (128, ), (1, ), 0), reinterpret_tensor(buf1416, (128, ), (1, ), 0), reinterpret_tensor(buf1407, (128, ), (1, ), 0), reinterpret_tensor(buf1406, (128, ), (1, ), 0), reinterpret_tensor(buf1397, (128, ), (1, ), 0), reinterpret_tensor(buf1396, (128, ), (1, ), 0), reinterpret_tensor(buf1387, (128, ), (1, ), 0), reinterpret_tensor(buf1386, (128, ), (1, ), 0), reinterpret_tensor(buf1382, (512, ), (1, ), 0), reinterpret_tensor(buf1380, (512, ), (1, ), 0), reinterpret_tensor(buf1375, (128, ), (1, ), 0), reinterpret_tensor(buf1374, (128, ), (1, ), 0), reinterpret_tensor(buf1369, (128, ), (1, ), 0), reinterpret_tensor(buf1368, (128, ), (1, ), 0), reinterpret_tensor(buf1350, (128, ), (1, ), 0), reinterpret_tensor(buf1349, (128, ), (1, ), 0), reinterpret_tensor(buf1340, (128, ), (1, ), 0), reinterpret_tensor(buf1339, (128, ), (1, ), 0), reinterpret_tensor(buf1330, (128, ), (1, ), 0), reinterpret_tensor(buf1329, (128, ), (1, ), 0), reinterpret_tensor(buf1320, (128, ), (1, ), 0), reinterpret_tensor(buf1319, (128, ), (1, ), 0), reinterpret_tensor(buf1310, (128, ), (1, ), 0), reinterpret_tensor(buf1309, (128, ), (1, ), 0), reinterpret_tensor(buf1305, (512, ), (1, ), 0), reinterpret_tensor(buf1303, (512, ), (1, ), 0), reinterpret_tensor(buf1298, (128, ), (1, ), 0), reinterpret_tensor(buf1297, (128, ), (1, ), 0), reinterpret_tensor(buf1292, (128, ), (1, ), 0), reinterpret_tensor(buf1291, (128, ), (1, ), 0), reinterpret_tensor(buf1273, (128, ), (1, ), 0), reinterpret_tensor(buf1272, (128, ), (1, ), 0), reinterpret_tensor(buf1263, (128, ), (1, ), 0), reinterpret_tensor(buf1262, (128, ), (1, ), 0), reinterpret_tensor(buf1253, (128, ), (1, ), 0), reinterpret_tensor(buf1252, (128, ), (1, ), 0), reinterpret_tensor(buf1243, (128, ), (1, ), 0), reinterpret_tensor(buf1242, (128, ), (1, ), 0), reinterpret_tensor(buf1233, (128, ), (1, ), 0), reinterpret_tensor(buf1232, (128, ), (1, ), 0), reinterpret_tensor(buf1228, (512, ), (1, ), 0), reinterpret_tensor(buf1226, (512, ), (1, ), 0), reinterpret_tensor(buf1221, (128, ), (1, ), 0), reinterpret_tensor(buf1220, (128, ), (1, ), 0), reinterpret_tensor(buf1215, (128, ), (1, ), 0), reinterpret_tensor(buf1214, (128, ), (1, ), 0), reinterpret_tensor(buf1196, (128, ), (1, ), 0), reinterpret_tensor(buf1195, (128, ), (1, ), 0), reinterpret_tensor(buf1186, (128, ), (1, ), 0), reinterpret_tensor(buf1185, (128, ), (1, ), 0), reinterpret_tensor(buf1176, (128, ), (1, ), 0), reinterpret_tensor(buf1175, (128, ), (1, ), 0), reinterpret_tensor(buf1166, (128, ), (1, ), 0), reinterpret_tensor(buf1165, (128, ), (1, ), 0), reinterpret_tensor(buf1156, (128, ), (1, ), 0), reinterpret_tensor(buf1155, (128, ), (1, ), 0), reinterpret_tensor(buf1151, (512, ), (1, ), 0), reinterpret_tensor(buf1149, (512, ), (1, ), 0), reinterpret_tensor(buf1144, (128, ), (1, ), 0), reinterpret_tensor(buf1143, (128, ), (1, ), 0), reinterpret_tensor(buf1138, (128, ), (1, ), 0), reinterpret_tensor(buf1137, (128, ), (1, ), 0), reinterpret_tensor(buf1119, (128, ), (1, ), 0), reinterpret_tensor(buf1118, (128, ), (1, ), 0), reinterpret_tensor(buf1109, (128, ), (1, ), 0), reinterpret_tensor(buf1108, (128, ), (1, ), 0), reinterpret_tensor(buf1099, (128, ), (1, ), 0), reinterpret_tensor(buf1098, (128, ), (1, ), 0), reinterpret_tensor(buf1089, (128, ), (1, ), 0), reinterpret_tensor(buf1088, (128, ), (1, ), 0), reinterpret_tensor(buf1079, (128, ), (1, ), 0), reinterpret_tensor(buf1078, (128, ), (1, ), 0), reinterpret_tensor(buf1074, (512, ), (1, ), 0), reinterpret_tensor(buf1072, (512, ), (1, ), 0), reinterpret_tensor(buf1067, (128, ), (1, ), 0), reinterpret_tensor(buf1066, (128, ), (1, ), 0), reinterpret_tensor(buf1061, (128, ), (1, ), 0), reinterpret_tensor(buf1060, (128, ), (1, ), 0), reinterpret_tensor(buf1042, (128, ), (1, ), 0), reinterpret_tensor(buf1041, (128, ), (1, ), 0), reinterpret_tensor(buf1032, (128, ), (1, ), 0), reinterpret_tensor(buf1031, (128, ), (1, ), 0), reinterpret_tensor(buf1022, (128, ), (1, ), 0), reinterpret_tensor(buf1021, (128, ), (1, ), 0), reinterpret_tensor(buf1012, (128, ), (1, ), 0), reinterpret_tensor(buf1011, (128, ), (1, ), 0), reinterpret_tensor(buf1002, (128, ), (1, ), 0), reinterpret_tensor(buf1001, (128, ), (1, ), 0), reinterpret_tensor(buf997, (512, ), (1, ), 0), reinterpret_tensor(buf995, (512, ), (1, ), 0), reinterpret_tensor(buf990, (128, ), (1, ), 0), reinterpret_tensor(buf989, (128, ), (1, ), 0), reinterpret_tensor(buf984, (128, ), (1, ), 0), reinterpret_tensor(buf983, (128, ), (1, ), 0), reinterpret_tensor(buf965, (128, ), (1, ), 0), reinterpret_tensor(buf964, (128, ), (1, ), 0), reinterpret_tensor(buf955, (128, ), (1, ), 0), reinterpret_tensor(buf954, (128, ), (1, ), 0), reinterpret_tensor(buf945, (128, ), (1, ), 0), reinterpret_tensor(buf944, (128, ), (1, ), 0), reinterpret_tensor(buf935, (128, ), (1, ), 0), reinterpret_tensor(buf934, (128, ), (1, ), 0), reinterpret_tensor(buf925, (128, ), (1, ), 0), reinterpret_tensor(buf924, (128, ), (1, ), 0), reinterpret_tensor(buf920, (512, ), (1, ), 0), reinterpret_tensor(buf918, (512, ), (1, ), 0), reinterpret_tensor(buf913, (128, ), (1, ), 0), reinterpret_tensor(buf912, (128, ), (1, ), 0), reinterpret_tensor(buf907, (128, ), (1, ), 0), reinterpret_tensor(buf906, (128, ), (1, ), 0), reinterpret_tensor(buf888, (128, ), (1, ), 0), reinterpret_tensor(buf887, (128, ), (1, ), 0), reinterpret_tensor(buf878, (128, ), (1, ), 0), reinterpret_tensor(buf877, (128, ), (1, ), 0), reinterpret_tensor(buf868, (128, ), (1, ), 0), reinterpret_tensor(buf867, (128, ), (1, ), 0), reinterpret_tensor(buf858, (128, ), (1, ), 0), reinterpret_tensor(buf857, (128, ), (1, ), 0), reinterpret_tensor(buf848, (128, ), (1, ), 0), reinterpret_tensor(buf847, (128, ), (1, ), 0), reinterpret_tensor(buf843, (512, ), (1, ), 0), reinterpret_tensor(buf841, (512, ), (1, ), 0), reinterpret_tensor(buf836, (128, ), (1, ), 0), reinterpret_tensor(buf835, (128, ), (1, ), 0), reinterpret_tensor(buf830, (128, ), (1, ), 0), reinterpret_tensor(buf829, (128, ), (1, ), 0), reinterpret_tensor(buf811, (128, ), (1, ), 0), reinterpret_tensor(buf810, (128, ), (1, ), 0), reinterpret_tensor(buf801, (128, ), (1, ), 0), reinterpret_tensor(buf800, (128, ), (1, ), 0), reinterpret_tensor(buf791, (128, ), (1, ), 0), reinterpret_tensor(buf790, (128, ), (1, ), 0), reinterpret_tensor(buf781, (128, ), (1, ), 0), reinterpret_tensor(buf780, (128, ), (1, ), 0), reinterpret_tensor(buf771, (128, ), (1, ), 0), reinterpret_tensor(buf770, (128, ), (1, ), 0), reinterpret_tensor(buf766, (512, ), (1, ), 0), reinterpret_tensor(buf764, (512, ), (1, ), 0), reinterpret_tensor(buf759, (128, ), (1, ), 0), reinterpret_tensor(buf758, (128, ), (1, ), 0), reinterpret_tensor(buf753, (128, ), (1, ), 0), reinterpret_tensor(buf752, (128, ), (1, ), 0), reinterpret_tensor(buf734, (128, ), (1, ), 0), reinterpret_tensor(buf733, (128, ), (1, ), 0), reinterpret_tensor(buf724, (128, ), (1, ), 0), reinterpret_tensor(buf723, (128, ), (1, ), 0), reinterpret_tensor(buf714, (128, ), (1, ), 0), reinterpret_tensor(buf713, (128, ), (1, ), 0), reinterpret_tensor(buf704, (128, ), (1, ), 0), reinterpret_tensor(buf703, (128, ), (1, ), 0), reinterpret_tensor(buf694, (128, ), (1, ), 0), reinterpret_tensor(buf693, (128, ), (1, ), 0), reinterpret_tensor(buf689, (512, ), (1, ), 0), reinterpret_tensor(buf687, (512, ), (1, ), 0), reinterpret_tensor(buf682, (128, ), (1, ), 0), reinterpret_tensor(buf681, (128, ), (1, ), 0), reinterpret_tensor(buf676, (128, ), (1, ), 0), reinterpret_tensor(buf675, (128, ), (1, ), 0), reinterpret_tensor(buf657, (128, ), (1, ), 0), reinterpret_tensor(buf656, (128, ), (1, ), 0), reinterpret_tensor(buf647, (128, ), (1, ), 0), reinterpret_tensor(buf646, (128, ), (1, ), 0), reinterpret_tensor(buf637, (128, ), (1, ), 0), reinterpret_tensor(buf636, (128, ), (1, ), 0), reinterpret_tensor(buf627, (128, ), (1, ), 0), reinterpret_tensor(buf626, (128, ), (1, ), 0), reinterpret_tensor(buf617, (128, ), (1, ), 0), reinterpret_tensor(buf616, (128, ), (1, ), 0), reinterpret_tensor(buf612, (512, ), (1, ), 0), reinterpret_tensor(buf610, (512, ), (1, ), 0), reinterpret_tensor(buf605, (128, ), (1, ), 0), reinterpret_tensor(buf604, (128, ), (1, ), 0), reinterpret_tensor(buf599, (128, ), (1, ), 0), reinterpret_tensor(buf598, (128, ), (1, ), 0), reinterpret_tensor(buf580, (128, ), (1, ), 0), reinterpret_tensor(buf579, (128, ), (1, ), 0), reinterpret_tensor(buf570, (128, ), (1, ), 0), reinterpret_tensor(buf569, (128, ), (1, ), 0), reinterpret_tensor(buf560, (128, ), (1, ), 0), reinterpret_tensor(buf559, (128, ), (1, ), 0), reinterpret_tensor(buf550, (128, ), (1, ), 0), reinterpret_tensor(buf549, (128, ), (1, ), 0), reinterpret_tensor(buf540, (128, ), (1, ), 0), reinterpret_tensor(buf539, (128, ), (1, ), 0), reinterpret_tensor(buf535, (512, ), (1, ), 0), reinterpret_tensor(buf533, (512, ), (1, ), 0), reinterpret_tensor(buf528, (128, ), (1, ), 0), reinterpret_tensor(buf527, (128, ), (1, ), 0), reinterpret_tensor(buf522, (128, ), (1, ), 0), reinterpret_tensor(buf521, (128, ), (1, ), 0), reinterpret_tensor(buf503, (128, ), (1, ), 0), reinterpret_tensor(buf502, (128, ), (1, ), 0), reinterpret_tensor(buf493, (128, ), (1, ), 0), reinterpret_tensor(buf492, (128, ), (1, ), 0), reinterpret_tensor(buf483, (128, ), (1, ), 0), reinterpret_tensor(buf482, (128, ), (1, ), 0), reinterpret_tensor(buf473, (128, ), (1, ), 0), reinterpret_tensor(buf472, (128, ), (1, ), 0), reinterpret_tensor(buf463, (128, ), (1, ), 0), reinterpret_tensor(buf462, (128, ), (1, ), 0), reinterpret_tensor(buf458, (512, ), (1, ), 0), reinterpret_tensor(buf456, (512, ), (1, ), 0), reinterpret_tensor(buf451, (128, ), (1, ), 0), reinterpret_tensor(buf450, (128, ), (1, ), 0), reinterpret_tensor(buf445, (128, ), (1, ), 0), reinterpret_tensor(buf444, (128, ), (1, ), 0), reinterpret_tensor(buf426, (128, ), (1, ), 0), reinterpret_tensor(buf425, (128, ), (1, ), 0), reinterpret_tensor(buf416, (128, ), (1, ), 0), reinterpret_tensor(buf415, (128, ), (1, ), 0), reinterpret_tensor(buf406, (128, ), (1, ), 0), reinterpret_tensor(buf405, (128, ), (1, ), 0), reinterpret_tensor(buf396, (128, ), (1, ), 0), reinterpret_tensor(buf395, (128, ), (1, ), 0), reinterpret_tensor(buf386, (128, ), (1, ), 0), reinterpret_tensor(buf385, (128, ), (1, ), 0), reinterpret_tensor(buf381, (512, ), (1, ), 0), reinterpret_tensor(buf379, (512, ), (1, ), 0), reinterpret_tensor(buf374, (128, ), (1, ), 0), reinterpret_tensor(buf373, (128, ), (1, ), 0), reinterpret_tensor(buf368, (128, ), (1, ), 0), reinterpret_tensor(buf367, (128, ), (1, ), 0), reinterpret_tensor(buf349, (128, ), (1, ), 0), reinterpret_tensor(buf348, (128, ), (1, ), 0), reinterpret_tensor(buf339, (128, ), (1, ), 0), reinterpret_tensor(buf338, (128, ), (1, ), 0), reinterpret_tensor(buf329, (128, ), (1, ), 0), reinterpret_tensor(buf328, (128, ), (1, ), 0), reinterpret_tensor(buf319, (128, ), (1, ), 0), reinterpret_tensor(buf318, (128, ), (1, ), 0), reinterpret_tensor(buf309, (128, ), (1, ), 0), reinterpret_tensor(buf308, (128, ), (1, ), 0), reinterpret_tensor(buf304, (512, ), (1, ), 0), reinterpret_tensor(buf302, (512, ), (1, ), 0), reinterpret_tensor(buf297, (128, ), (1, ), 0), reinterpret_tensor(buf296, (128, ), (1, ), 0), reinterpret_tensor(buf291, (128, ), (1, ), 0), reinterpret_tensor(buf290, (128, ), (1, ), 0), reinterpret_tensor(buf272, (128, ), (1, ), 0), reinterpret_tensor(buf271, (128, ), (1, ), 0), reinterpret_tensor(buf262, (128, ), (1, ), 0), reinterpret_tensor(buf261, (128, ), (1, ), 0), reinterpret_tensor(buf252, (128, ), (1, ), 0), reinterpret_tensor(buf251, (128, ), (1, ), 0), reinterpret_tensor(buf242, (128, ), (1, ), 0), reinterpret_tensor(buf241, (128, ), (1, ), 0), reinterpret_tensor(buf232, (128, ), (1, ), 0), reinterpret_tensor(buf231, (128, ), (1, ), 0), reinterpret_tensor(buf227, (512, ), (1, ), 0), reinterpret_tensor(buf225, (512, ), (1, ), 0), reinterpret_tensor(buf220, (128, ), (1, ), 0), reinterpret_tensor(buf219, (128, ), (1, ), 0), reinterpret_tensor(buf214, (128, ), (1, ), 0), reinterpret_tensor(buf213, (128, ), (1, ), 0), reinterpret_tensor(buf195, (128, ), (1, ), 0), reinterpret_tensor(buf194, (128, ), (1, ), 0), reinterpret_tensor(buf185, (128, ), (1, ), 0), reinterpret_tensor(buf184, (128, ), (1, ), 0), reinterpret_tensor(buf175, (128, ), (1, ), 0), reinterpret_tensor(buf174, (128, ), (1, ), 0), reinterpret_tensor(buf165, (128, ), (1, ), 0), reinterpret_tensor(buf164, (128, ), (1, ), 0), reinterpret_tensor(buf155, (128, ), (1, ), 0), reinterpret_tensor(buf154, (128, ), (1, ), 0), reinterpret_tensor(buf150, (512, ), (1, ), 0), reinterpret_tensor(buf148, (512, ), (1, ), 0), reinterpret_tensor(buf143, (128, ), (1, ), 0), reinterpret_tensor(buf142, (128, ), (1, ), 0), reinterpret_tensor(buf137, (128, ), (1, ), 0), reinterpret_tensor(buf136, (128, ), (1, ), 0), reinterpret_tensor(buf118, (128, ), (1, ), 0), reinterpret_tensor(buf117, (128, ), (1, ), 0), reinterpret_tensor(buf108, (128, ), (1, ), 0), reinterpret_tensor(buf107, (128, ), (1, ), 0), reinterpret_tensor(buf98, (128, ), (1, ), 0), reinterpret_tensor(buf97, (128, ), (1, ), 0), reinterpret_tensor(buf88, (128, ), (1, ), 0), reinterpret_tensor(buf87, (128, ), (1, ), 0), reinterpret_tensor(buf78, (128, ), (1, ), 0), reinterpret_tensor(buf77, (128, ), (1, ), 0), reinterpret_tensor(buf72, (512, ), (1, ), 0), reinterpret_tensor(buf71, (512, ), (1, ), 0), buf1933, reinterpret_tensor(buf1931, (512, 384), (384, 1), 0), reinterpret_tensor(buf1932, (512, ), (1, ), 0), buf1926, buf1922, reinterpret_tensor(buf1917, (128, 512), (512, 1), 0), reinterpret_tensor(buf1918, (128, ), (1, ), 0), reinterpret_tensor(buf1911, (128, 512), (512, 1), 0), reinterpret_tensor(buf1912, (128, ), (1, ), 0), reinterpret_tensor(buf1905, (128, 128), (128, 1), 0), reinterpret_tensor(buf1906, (128, ), (1, ), 0), reinterpret_tensor(buf1902, (128, 128), (128, 1), 0), reinterpret_tensor(buf1903, (128, ), (1, ), 0), reinterpret_tensor(buf1899, (128, 512), (512, 1), 0), reinterpret_tensor(buf1900, (128, ), (1, ), 0), reinterpret_tensor(buf1892, (128, 128), (128, 1), 0), reinterpret_tensor(buf1893, (128, ), (1, ), 0), reinterpret_tensor(buf1886, (512, 128), (128, 1), 0), reinterpret_tensor(buf1887, (512, ), (1, ), 0), reinterpret_tensor(buf1882, (128, 512), (512, 1), 0), reinterpret_tensor(buf1883, (128, ), (1, ), 0), reinterpret_tensor(buf1876, (512, 128), (128, 1), 0), reinterpret_tensor(buf1877, (512, ), (1, ), 0), reinterpret_tensor(buf1872, (128, 512), (512, 1), 0), reinterpret_tensor(buf1873, (128, ), (1, ), 0), reinterpret_tensor(buf1866, (512, 128), (128, 1), 0), reinterpret_tensor(buf1867, (512, ), (1, ), 0), reinterpret_tensor(buf1862, (128, 512), (512, 1), 0), reinterpret_tensor(buf1863, (128, ), (1, ), 0), reinterpret_tensor(buf1856, (512, 128), (128, 1), 0), reinterpret_tensor(buf1857, (512, ), (1, ), 0), reinterpret_tensor(buf1852, (128, 512), (512, 1), 0), reinterpret_tensor(buf1853, (128, ), (1, ), 0), reinterpret_tensor(buf1846, (512, 128), (128, 1), 0), reinterpret_tensor(buf1847, (512, ), (1, ), 0), reinterpret_tensor(buf1840, (128, 512), (512, 1), 0), reinterpret_tensor(buf1841, (128, ), (1, ), 0), reinterpret_tensor(buf1834, (128, 512), (512, 1), 0), reinterpret_tensor(buf1835, (128, ), (1, ), 0), reinterpret_tensor(buf1828, (128, 128), (128, 1), 0), reinterpret_tensor(buf1829, (128, ), (1, ), 0), reinterpret_tensor(buf1825, (128, 128), (128, 1), 0), reinterpret_tensor(buf1826, (128, ), (1, ), 0), reinterpret_tensor(buf1822, (128, 512), (512, 1), 0), reinterpret_tensor(buf1823, (128, ), (1, ), 0), reinterpret_tensor(buf1815, (128, 128), (128, 1), 0), reinterpret_tensor(buf1816, (128, ), (1, ), 0), reinterpret_tensor(buf1809, (512, 128), (128, 1), 0), reinterpret_tensor(buf1810, (512, ), (1, ), 0), reinterpret_tensor(buf1805, (128, 512), (512, 1), 0), reinterpret_tensor(buf1806, (128, ), (1, ), 0), reinterpret_tensor(buf1799, (512, 128), (128, 1), 0), reinterpret_tensor(buf1800, (512, ), (1, ), 0), reinterpret_tensor(buf1795, (128, 512), (512, 1), 0), reinterpret_tensor(buf1796, (128, ), (1, ), 0), reinterpret_tensor(buf1789, (512, 128), (128, 1), 0), reinterpret_tensor(buf1790, (512, ), (1, ), 0), reinterpret_tensor(buf1785, (128, 512), (512, 1), 0), reinterpret_tensor(buf1786, (128, ), (1, ), 0), reinterpret_tensor(buf1779, (512, 128), (128, 1), 0), reinterpret_tensor(buf1780, (512, ), (1, ), 0), reinterpret_tensor(buf1775, (128, 512), (512, 1), 0), reinterpret_tensor(buf1776, (128, ), (1, ), 0), reinterpret_tensor(buf1769, (512, 128), (128, 1), 0), reinterpret_tensor(buf1770, (512, ), (1, ), 0), reinterpret_tensor(buf1763, (128, 512), (512, 1), 0), reinterpret_tensor(buf1764, (128, ), (1, ), 0), reinterpret_tensor(buf1757, (128, 512), (512, 1), 0), reinterpret_tensor(buf1758, (128, ), (1, ), 0), reinterpret_tensor(buf1751, (128, 128), (128, 1), 0), reinterpret_tensor(buf1752, (128, ), (1, ), 0), reinterpret_tensor(buf1748, (128, 128), (128, 1), 0), reinterpret_tensor(buf1749, (128, ), (1, ), 0), reinterpret_tensor(buf1745, (128, 512), (512, 1), 0), reinterpret_tensor(buf1746, (128, ), (1, ), 0), reinterpret_tensor(buf1738, (128, 128), (128, 1), 0), reinterpret_tensor(buf1739, (128, ), (1, ), 0), reinterpret_tensor(buf1732, (512, 128), (128, 1), 0), reinterpret_tensor(buf1733, (512, ), (1, ), 0), reinterpret_tensor(buf1728, (128, 512), (512, 1), 0), reinterpret_tensor(buf1729, (128, ), (1, ), 0), reinterpret_tensor(buf1722, (512, 128), (128, 1), 0), reinterpret_tensor(buf1723, (512, ), (1, ), 0), reinterpret_tensor(buf1718, (128, 512), (512, 1), 0), reinterpret_tensor(buf1719, (128, ), (1, ), 0), reinterpret_tensor(buf1712, (512, 128), (128, 1), 0), reinterpret_tensor(buf1713, (512, ), (1, ), 0), reinterpret_tensor(buf1708, (128, 512), (512, 1), 0), reinterpret_tensor(buf1709, (128, ), (1, ), 0), reinterpret_tensor(buf1702, (512, 128), (128, 1), 0), reinterpret_tensor(buf1703, (512, ), (1, ), 0), reinterpret_tensor(buf1698, (128, 512), (512, 1), 0), reinterpret_tensor(buf1699, (128, ), (1, ), 0), reinterpret_tensor(buf1692, (512, 128), (128, 1), 0), reinterpret_tensor(buf1693, (512, ), (1, ), 0), reinterpret_tensor(buf1686, (128, 512), (512, 1), 0), reinterpret_tensor(buf1687, (128, ), (1, ), 0), reinterpret_tensor(buf1680, (128, 512), (512, 1), 0), reinterpret_tensor(buf1681, (128, ), (1, ), 0), reinterpret_tensor(buf1674, (128, 128), (128, 1), 0), reinterpret_tensor(buf1675, (128, ), (1, ), 0), reinterpret_tensor(buf1671, (128, 128), (128, 1), 0), reinterpret_tensor(buf1672, (128, ), (1, ), 0), reinterpret_tensor(buf1668, (128, 512), (512, 1), 0), reinterpret_tensor(buf1669, (128, ), (1, ), 0), reinterpret_tensor(buf1661, (128, 128), (128, 1), 0), reinterpret_tensor(buf1662, (128, ), (1, ), 0), reinterpret_tensor(buf1655, (512, 128), (128, 1), 0), reinterpret_tensor(buf1656, (512, ), (1, ), 0), reinterpret_tensor(buf1651, (128, 512), (512, 1), 0), reinterpret_tensor(buf1652, (128, ), (1, ), 0), reinterpret_tensor(buf1645, (512, 128), (128, 1), 0), reinterpret_tensor(buf1646, (512, ), (1, ), 0), reinterpret_tensor(buf1641, (128, 512), (512, 1), 0), reinterpret_tensor(buf1642, (128, ), (1, ), 0), reinterpret_tensor(buf1635, (512, 128), (128, 1), 0), reinterpret_tensor(buf1636, (512, ), (1, ), 0), reinterpret_tensor(buf1631, (128, 512), (512, 1), 0), reinterpret_tensor(buf1632, (128, ), (1, ), 0), reinterpret_tensor(buf1625, (512, 128), (128, 1), 0), reinterpret_tensor(buf1626, (512, ), (1, ), 0), reinterpret_tensor(buf1621, (128, 512), (512, 1), 0), reinterpret_tensor(buf1622, (128, ), (1, ), 0), reinterpret_tensor(buf1615, (512, 128), (128, 1), 0), reinterpret_tensor(buf1616, (512, ), (1, ), 0), reinterpret_tensor(buf1609, (128, 512), (512, 1), 0), reinterpret_tensor(buf1610, (128, ), (1, ), 0), reinterpret_tensor(buf1603, (128, 512), (512, 1), 0), reinterpret_tensor(buf1604, (128, ), (1, ), 0), reinterpret_tensor(buf1597, (128, 128), (128, 1), 0), reinterpret_tensor(buf1598, (128, ), (1, ), 0), reinterpret_tensor(buf1594, (128, 128), (128, 1), 0), reinterpret_tensor(buf1595, (128, ), (1, ), 0), reinterpret_tensor(buf1591, (128, 512), (512, 1), 0), reinterpret_tensor(buf1592, (128, ), (1, ), 0), reinterpret_tensor(buf1584, (128, 128), (128, 1), 0), reinterpret_tensor(buf1585, (128, ), (1, ), 0), reinterpret_tensor(buf1578, (512, 128), (128, 1), 0), reinterpret_tensor(buf1579, (512, ), (1, ), 0), reinterpret_tensor(buf1574, (128, 512), (512, 1), 0), reinterpret_tensor(buf1575, (128, ), (1, ), 0), reinterpret_tensor(buf1568, (512, 128), (128, 1), 0), reinterpret_tensor(buf1569, (512, ), (1, ), 0), reinterpret_tensor(buf1564, (128, 512), (512, 1), 0), reinterpret_tensor(buf1565, (128, ), (1, ), 0), reinterpret_tensor(buf1558, (512, 128), (128, 1), 0), reinterpret_tensor(buf1559, (512, ), (1, ), 0), reinterpret_tensor(buf1554, (128, 512), (512, 1), 0), reinterpret_tensor(buf1555, (128, ), (1, ), 0), reinterpret_tensor(buf1548, (512, 128), (128, 1), 0), reinterpret_tensor(buf1549, (512, ), (1, ), 0), reinterpret_tensor(buf1544, (128, 512), (512, 1), 0), reinterpret_tensor(buf1545, (128, ), (1, ), 0), reinterpret_tensor(buf1538, (512, 128), (128, 1), 0), reinterpret_tensor(buf1539, (512, ), (1, ), 0), reinterpret_tensor(buf1532, (128, 512), (512, 1), 0), reinterpret_tensor(buf1533, (128, ), (1, ), 0), reinterpret_tensor(buf1526, (128, 512), (512, 1), 0), reinterpret_tensor(buf1527, (128, ), (1, ), 0), reinterpret_tensor(buf1520, (128, 128), (128, 1), 0), reinterpret_tensor(buf1521, (128, ), (1, ), 0), reinterpret_tensor(buf1517, (128, 128), (128, 1), 0), reinterpret_tensor(buf1518, (128, ), (1, ), 0), reinterpret_tensor(buf1514, (128, 512), (512, 1), 0), reinterpret_tensor(buf1515, (128, ), (1, ), 0), reinterpret_tensor(buf1507, (128, 128), (128, 1), 0), reinterpret_tensor(buf1508, (128, ), (1, ), 0), reinterpret_tensor(buf1501, (512, 128), (128, 1), 0), reinterpret_tensor(buf1502, (512, ), (1, ), 0), reinterpret_tensor(buf1497, (128, 512), (512, 1), 0), reinterpret_tensor(buf1498, (128, ), (1, ), 0), reinterpret_tensor(buf1491, (512, 128), (128, 1), 0), reinterpret_tensor(buf1492, (512, ), (1, ), 0), reinterpret_tensor(buf1487, (128, 512), (512, 1), 0), reinterpret_tensor(buf1488, (128, ), (1, ), 0), reinterpret_tensor(buf1481, (512, 128), (128, 1), 0), reinterpret_tensor(buf1482, (512, ), (1, ), 0), reinterpret_tensor(buf1477, (128, 512), (512, 1), 0), reinterpret_tensor(buf1478, (128, ), (1, ), 0), reinterpret_tensor(buf1471, (512, 128), (128, 1), 0), reinterpret_tensor(buf1472, (512, ), (1, ), 0), reinterpret_tensor(buf1467, (128, 512), (512, 1), 0), reinterpret_tensor(buf1468, (128, ), (1, ), 0), reinterpret_tensor(buf1461, (512, 128), (128, 1), 0), reinterpret_tensor(buf1462, (512, ), (1, ), 0), reinterpret_tensor(buf1455, (128, 512), (512, 1), 0), reinterpret_tensor(buf1456, (128, ), (1, ), 0), reinterpret_tensor(buf1449, (128, 512), (512, 1), 0), reinterpret_tensor(buf1450, (128, ), (1, ), 0), reinterpret_tensor(buf1443, (128, 128), (128, 1), 0), reinterpret_tensor(buf1444, (128, ), (1, ), 0), reinterpret_tensor(buf1440, (128, 128), (128, 1), 0), reinterpret_tensor(buf1441, (128, ), (1, ), 0), reinterpret_tensor(buf1437, (128, 512), (512, 1), 0), reinterpret_tensor(buf1438, (128, ), (1, ), 0), reinterpret_tensor(buf1430, (128, 128), (128, 1), 0), reinterpret_tensor(buf1431, (128, ), (1, ), 0), reinterpret_tensor(buf1424, (512, 128), (128, 1), 0), reinterpret_tensor(buf1425, (512, ), (1, ), 0), reinterpret_tensor(buf1420, (128, 512), (512, 1), 0), reinterpret_tensor(buf1421, (128, ), (1, ), 0), reinterpret_tensor(buf1414, (512, 128), (128, 1), 0), reinterpret_tensor(buf1415, (512, ), (1, ), 0), reinterpret_tensor(buf1410, (128, 512), (512, 1), 0), reinterpret_tensor(buf1411, (128, ), (1, ), 0), reinterpret_tensor(buf1404, (512, 128), (128, 1), 0), reinterpret_tensor(buf1405, (512, ), (1, ), 0), reinterpret_tensor(buf1400, (128, 512), (512, 1), 0), reinterpret_tensor(buf1401, (128, ), (1, ), 0), reinterpret_tensor(buf1394, (512, 128), (128, 1), 0), reinterpret_tensor(buf1395, (512, ), (1, ), 0), reinterpret_tensor(buf1390, (128, 512), (512, 1), 0), reinterpret_tensor(buf1391, (128, ), (1, ), 0), reinterpret_tensor(buf1384, (512, 128), (128, 1), 0), reinterpret_tensor(buf1385, (512, ), (1, ), 0), reinterpret_tensor(buf1378, (128, 512), (512, 1), 0), reinterpret_tensor(buf1379, (128, ), (1, ), 0), reinterpret_tensor(buf1372, (128, 512), (512, 1), 0), reinterpret_tensor(buf1373, (128, ), (1, ), 0), reinterpret_tensor(buf1366, (128, 128), (128, 1), 0), reinterpret_tensor(buf1367, (128, ), (1, ), 0), reinterpret_tensor(buf1363, (128, 128), (128, 1), 0), reinterpret_tensor(buf1364, (128, ), (1, ), 0), reinterpret_tensor(buf1360, (128, 512), (512, 1), 0), reinterpret_tensor(buf1361, (128, ), (1, ), 0), reinterpret_tensor(buf1353, (128, 128), (128, 1), 0), reinterpret_tensor(buf1354, (128, ), (1, ), 0), reinterpret_tensor(buf1347, (512, 128), (128, 1), 0), reinterpret_tensor(buf1348, (512, ), (1, ), 0), reinterpret_tensor(buf1343, (128, 512), (512, 1), 0), reinterpret_tensor(buf1344, (128, ), (1, ), 0), reinterpret_tensor(buf1337, (512, 128), (128, 1), 0), reinterpret_tensor(buf1338, (512, ), (1, ), 0), reinterpret_tensor(buf1333, (128, 512), (512, 1), 0), reinterpret_tensor(buf1334, (128, ), (1, ), 0), reinterpret_tensor(buf1327, (512, 128), (128, 1), 0), reinterpret_tensor(buf1328, (512, ), (1, ), 0), reinterpret_tensor(buf1323, (128, 512), (512, 1), 0), reinterpret_tensor(buf1324, (128, ), (1, ), 0), reinterpret_tensor(buf1317, (512, 128), (128, 1), 0), reinterpret_tensor(buf1318, (512, ), (1, ), 0), reinterpret_tensor(buf1313, (128, 512), (512, 1), 0), reinterpret_tensor(buf1314, (128, ), (1, ), 0), reinterpret_tensor(buf1307, (512, 128), (128, 1), 0), reinterpret_tensor(buf1308, (512, ), (1, ), 0), reinterpret_tensor(buf1301, (128, 512), (512, 1), 0), reinterpret_tensor(buf1302, (128, ), (1, ), 0), reinterpret_tensor(buf1295, (128, 512), (512, 1), 0), reinterpret_tensor(buf1296, (128, ), (1, ), 0), reinterpret_tensor(buf1289, (128, 128), (128, 1), 0), reinterpret_tensor(buf1290, (128, ), (1, ), 0), reinterpret_tensor(buf1286, (128, 128), (128, 1), 0), reinterpret_tensor(buf1287, (128, ), (1, ), 0), reinterpret_tensor(buf1283, (128, 512), (512, 1), 0), reinterpret_tensor(buf1284, (128, ), (1, ), 0), reinterpret_tensor(buf1276, (128, 128), (128, 1), 0), reinterpret_tensor(buf1277, (128, ), (1, ), 0), reinterpret_tensor(buf1270, (512, 128), (128, 1), 0), reinterpret_tensor(buf1271, (512, ), (1, ), 0), reinterpret_tensor(buf1266, (128, 512), (512, 1), 0), reinterpret_tensor(buf1267, (128, ), (1, ), 0), reinterpret_tensor(buf1260, (512, 128), (128, 1), 0), reinterpret_tensor(buf1261, (512, ), (1, ), 0), reinterpret_tensor(buf1256, (128, 512), (512, 1), 0), reinterpret_tensor(buf1257, (128, ), (1, ), 0), reinterpret_tensor(buf1250, (512, 128), (128, 1), 0), reinterpret_tensor(buf1251, (512, ), (1, ), 0), reinterpret_tensor(buf1246, (128, 512), (512, 1), 0), reinterpret_tensor(buf1247, (128, ), (1, ), 0), reinterpret_tensor(buf1240, (512, 128), (128, 1), 0), reinterpret_tensor(buf1241, (512, ), (1, ), 0), reinterpret_tensor(buf1236, (128, 512), (512, 1), 0), reinterpret_tensor(buf1237, (128, ), (1, ), 0), reinterpret_tensor(buf1230, (512, 128), (128, 1), 0), reinterpret_tensor(buf1231, (512, ), (1, ), 0), reinterpret_tensor(buf1224, (128, 512), (512, 1), 0), reinterpret_tensor(buf1225, (128, ), (1, ), 0), reinterpret_tensor(buf1218, (128, 512), (512, 1), 0), reinterpret_tensor(buf1219, (128, ), (1, ), 0), reinterpret_tensor(buf1212, (128, 128), (128, 1), 0), reinterpret_tensor(buf1213, (128, ), (1, ), 0), reinterpret_tensor(buf1209, (128, 128), (128, 1), 0), reinterpret_tensor(buf1210, (128, ), (1, ), 0), reinterpret_tensor(buf1206, (128, 512), (512, 1), 0), reinterpret_tensor(buf1207, (128, ), (1, ), 0), reinterpret_tensor(buf1199, (128, 128), (128, 1), 0), reinterpret_tensor(buf1200, (128, ), (1, ), 0), reinterpret_tensor(buf1193, (512, 128), (128, 1), 0), reinterpret_tensor(buf1194, (512, ), (1, ), 0), reinterpret_tensor(buf1189, (128, 512), (512, 1), 0), reinterpret_tensor(buf1190, (128, ), (1, ), 0), reinterpret_tensor(buf1183, (512, 128), (128, 1), 0), reinterpret_tensor(buf1184, (512, ), (1, ), 0), reinterpret_tensor(buf1179, (128, 512), (512, 1), 0), reinterpret_tensor(buf1180, (128, ), (1, ), 0), reinterpret_tensor(buf1173, (512, 128), (128, 1), 0), reinterpret_tensor(buf1174, (512, ), (1, ), 0), reinterpret_tensor(buf1169, (128, 512), (512, 1), 0), reinterpret_tensor(buf1170, (128, ), (1, ), 0), reinterpret_tensor(buf1163, (512, 128), (128, 1), 0), reinterpret_tensor(buf1164, (512, ), (1, ), 0), reinterpret_tensor(buf1159, (128, 512), (512, 1), 0), reinterpret_tensor(buf1160, (128, ), (1, ), 0), reinterpret_tensor(buf1153, (512, 128), (128, 1), 0), reinterpret_tensor(buf1154, (512, ), (1, ), 0), reinterpret_tensor(buf1147, (128, 512), (512, 1), 0), reinterpret_tensor(buf1148, (128, ), (1, ), 0), reinterpret_tensor(buf1141, (128, 512), (512, 1), 0), reinterpret_tensor(buf1142, (128, ), (1, ), 0), reinterpret_tensor(buf1135, (128, 128), (128, 1), 0), reinterpret_tensor(buf1136, (128, ), (1, ), 0), reinterpret_tensor(buf1132, (128, 128), (128, 1), 0), reinterpret_tensor(buf1133, (128, ), (1, ), 0), reinterpret_tensor(buf1129, (128, 512), (512, 1), 0), reinterpret_tensor(buf1130, (128, ), (1, ), 0), reinterpret_tensor(buf1122, (128, 128), (128, 1), 0), reinterpret_tensor(buf1123, (128, ), (1, ), 0), reinterpret_tensor(buf1116, (512, 128), (128, 1), 0), reinterpret_tensor(buf1117, (512, ), (1, ), 0), reinterpret_tensor(buf1112, (128, 512), (512, 1), 0), reinterpret_tensor(buf1113, (128, ), (1, ), 0), reinterpret_tensor(buf1106, (512, 128), (128, 1), 0), reinterpret_tensor(buf1107, (512, ), (1, ), 0), reinterpret_tensor(buf1102, (128, 512), (512, 1), 0), reinterpret_tensor(buf1103, (128, ), (1, ), 0), reinterpret_tensor(buf1096, (512, 128), (128, 1), 0), reinterpret_tensor(buf1097, (512, ), (1, ), 0), reinterpret_tensor(buf1092, (128, 512), (512, 1), 0), reinterpret_tensor(buf1093, (128, ), (1, ), 0), reinterpret_tensor(buf1086, (512, 128), (128, 1), 0), reinterpret_tensor(buf1087, (512, ), (1, ), 0), reinterpret_tensor(buf1082, (128, 512), (512, 1), 0), reinterpret_tensor(buf1083, (128, ), (1, ), 0), reinterpret_tensor(buf1076, (512, 128), (128, 1), 0), reinterpret_tensor(buf1077, (512, ), (1, ), 0), reinterpret_tensor(buf1070, (128, 512), (512, 1), 0), reinterpret_tensor(buf1071, (128, ), (1, ), 0), reinterpret_tensor(buf1064, (128, 512), (512, 1), 0), reinterpret_tensor(buf1065, (128, ), (1, ), 0), reinterpret_tensor(buf1058, (128, 128), (128, 1), 0), reinterpret_tensor(buf1059, (128, ), (1, ), 0), reinterpret_tensor(buf1055, (128, 128), (128, 1), 0), reinterpret_tensor(buf1056, (128, ), (1, ), 0), reinterpret_tensor(buf1052, (128, 512), (512, 1), 0), reinterpret_tensor(buf1053, (128, ), (1, ), 0), reinterpret_tensor(buf1045, (128, 128), (128, 1), 0), reinterpret_tensor(buf1046, (128, ), (1, ), 0), reinterpret_tensor(buf1039, (512, 128), (128, 1), 0), reinterpret_tensor(buf1040, (512, ), (1, ), 0), reinterpret_tensor(buf1035, (128, 512), (512, 1), 0), reinterpret_tensor(buf1036, (128, ), (1, ), 0), reinterpret_tensor(buf1029, (512, 128), (128, 1), 0), reinterpret_tensor(buf1030, (512, ), (1, ), 0), reinterpret_tensor(buf1025, (128, 512), (512, 1), 0), reinterpret_tensor(buf1026, (128, ), (1, ), 0), reinterpret_tensor(buf1019, (512, 128), (128, 1), 0), reinterpret_tensor(buf1020, (512, ), (1, ), 0), reinterpret_tensor(buf1015, (128, 512), (512, 1), 0), reinterpret_tensor(buf1016, (128, ), (1, ), 0), reinterpret_tensor(buf1009, (512, 128), (128, 1), 0), reinterpret_tensor(buf1010, (512, ), (1, ), 0), reinterpret_tensor(buf1005, (128, 512), (512, 1), 0), reinterpret_tensor(buf1006, (128, ), (1, ), 0), reinterpret_tensor(buf999, (512, 128), (128, 1), 0), reinterpret_tensor(buf1000, (512, ), (1, ), 0), reinterpret_tensor(buf993, (128, 512), (512, 1), 0), reinterpret_tensor(buf994, (128, ), (1, ), 0), reinterpret_tensor(buf987, (128, 512), (512, 1), 0), reinterpret_tensor(buf988, (128, ), (1, ), 0), reinterpret_tensor(buf981, (128, 128), (128, 1), 0), reinterpret_tensor(buf982, (128, ), (1, ), 0), reinterpret_tensor(buf978, (128, 128), (128, 1), 0), reinterpret_tensor(buf979, (128, ), (1, ), 0), reinterpret_tensor(buf975, (128, 512), (512, 1), 0), reinterpret_tensor(buf976, (128, ), (1, ), 0), reinterpret_tensor(buf968, (128, 128), (128, 1), 0), reinterpret_tensor(buf969, (128, ), (1, ), 0), reinterpret_tensor(buf962, (512, 128), (128, 1), 0), reinterpret_tensor(buf963, (512, ), (1, ), 0), reinterpret_tensor(buf958, (128, 512), (512, 1), 0), reinterpret_tensor(buf959, (128, ), (1, ), 0), reinterpret_tensor(buf952, (512, 128), (128, 1), 0), reinterpret_tensor(buf953, (512, ), (1, ), 0), reinterpret_tensor(buf948, (128, 512), (512, 1), 0), reinterpret_tensor(buf949, (128, ), (1, ), 0), reinterpret_tensor(buf942, (512, 128), (128, 1), 0), reinterpret_tensor(buf943, (512, ), (1, ), 0), reinterpret_tensor(buf938, (128, 512), (512, 1), 0), reinterpret_tensor(buf939, (128, ), (1, ), 0), reinterpret_tensor(buf932, (512, 128), (128, 1), 0), reinterpret_tensor(buf933, (512, ), (1, ), 0), reinterpret_tensor(buf928, (128, 512), (512, 1), 0), reinterpret_tensor(buf929, (128, ), (1, ), 0), reinterpret_tensor(buf922, (512, 128), (128, 1), 0), reinterpret_tensor(buf923, (512, ), (1, ), 0), reinterpret_tensor(buf916, (128, 512), (512, 1), 0), reinterpret_tensor(buf917, (128, ), (1, ), 0), reinterpret_tensor(buf910, (128, 512), (512, 1), 0), reinterpret_tensor(buf911, (128, ), (1, ), 0), reinterpret_tensor(buf904, (128, 128), (128, 1), 0), reinterpret_tensor(buf905, (128, ), (1, ), 0), reinterpret_tensor(buf901, (128, 128), (128, 1), 0), reinterpret_tensor(buf902, (128, ), (1, ), 0), reinterpret_tensor(buf898, (128, 512), (512, 1), 0), reinterpret_tensor(buf899, (128, ), (1, ), 0), reinterpret_tensor(buf891, (128, 128), (128, 1), 0), reinterpret_tensor(buf892, (128, ), (1, ), 0), reinterpret_tensor(buf885, (512, 128), (128, 1), 0), reinterpret_tensor(buf886, (512, ), (1, ), 0), reinterpret_tensor(buf881, (128, 512), (512, 1), 0), reinterpret_tensor(buf882, (128, ), (1, ), 0), reinterpret_tensor(buf875, (512, 128), (128, 1), 0), reinterpret_tensor(buf876, (512, ), (1, ), 0), reinterpret_tensor(buf871, (128, 512), (512, 1), 0), reinterpret_tensor(buf872, (128, ), (1, ), 0), reinterpret_tensor(buf865, (512, 128), (128, 1), 0), reinterpret_tensor(buf866, (512, ), (1, ), 0), reinterpret_tensor(buf861, (128, 512), (512, 1), 0), reinterpret_tensor(buf862, (128, ), (1, ), 0), reinterpret_tensor(buf855, (512, 128), (128, 1), 0), reinterpret_tensor(buf856, (512, ), (1, ), 0), reinterpret_tensor(buf851, (128, 512), (512, 1), 0), reinterpret_tensor(buf852, (128, ), (1, ), 0), reinterpret_tensor(buf845, (512, 128), (128, 1), 0), reinterpret_tensor(buf846, (512, ), (1, ), 0), reinterpret_tensor(buf839, (128, 512), (512, 1), 0), reinterpret_tensor(buf840, (128, ), (1, ), 0), reinterpret_tensor(buf833, (128, 512), (512, 1), 0), reinterpret_tensor(buf834, (128, ), (1, ), 0), reinterpret_tensor(buf827, (128, 128), (128, 1), 0), reinterpret_tensor(buf828, (128, ), (1, ), 0), reinterpret_tensor(buf824, (128, 128), (128, 1), 0), reinterpret_tensor(buf825, (128, ), (1, ), 0), reinterpret_tensor(buf821, (128, 512), (512, 1), 0), reinterpret_tensor(buf822, (128, ), (1, ), 0), reinterpret_tensor(buf814, (128, 128), (128, 1), 0), reinterpret_tensor(buf815, (128, ), (1, ), 0), reinterpret_tensor(buf808, (512, 128), (128, 1), 0), reinterpret_tensor(buf809, (512, ), (1, ), 0), reinterpret_tensor(buf804, (128, 512), (512, 1), 0), reinterpret_tensor(buf805, (128, ), (1, ), 0), reinterpret_tensor(buf798, (512, 128), (128, 1), 0), reinterpret_tensor(buf799, (512, ), (1, ), 0), reinterpret_tensor(buf794, (128, 512), (512, 1), 0), reinterpret_tensor(buf795, (128, ), (1, ), 0), reinterpret_tensor(buf788, (512, 128), (128, 1), 0), reinterpret_tensor(buf789, (512, ), (1, ), 0), reinterpret_tensor(buf784, (128, 512), (512, 1), 0), reinterpret_tensor(buf785, (128, ), (1, ), 0), reinterpret_tensor(buf778, (512, 128), (128, 1), 0), reinterpret_tensor(buf779, (512, ), (1, ), 0), reinterpret_tensor(buf774, (128, 512), (512, 1), 0), reinterpret_tensor(buf775, (128, ), (1, ), 0), reinterpret_tensor(buf768, (512, 128), (128, 1), 0), reinterpret_tensor(buf769, (512, ), (1, ), 0), reinterpret_tensor(buf762, (128, 512), (512, 1), 0), reinterpret_tensor(buf763, (128, ), (1, ), 0), reinterpret_tensor(buf756, (128, 512), (512, 1), 0), reinterpret_tensor(buf757, (128, ), (1, ), 0), reinterpret_tensor(buf750, (128, 128), (128, 1), 0), reinterpret_tensor(buf751, (128, ), (1, ), 0), reinterpret_tensor(buf747, (128, 128), (128, 1), 0), reinterpret_tensor(buf748, (128, ), (1, ), 0), reinterpret_tensor(buf744, (128, 512), (512, 1), 0), reinterpret_tensor(buf745, (128, ), (1, ), 0), reinterpret_tensor(buf737, (128, 128), (128, 1), 0), reinterpret_tensor(buf738, (128, ), (1, ), 0), reinterpret_tensor(buf731, (512, 128), (128, 1), 0), reinterpret_tensor(buf732, (512, ), (1, ), 0), reinterpret_tensor(buf727, (128, 512), (512, 1), 0), reinterpret_tensor(buf728, (128, ), (1, ), 0), reinterpret_tensor(buf721, (512, 128), (128, 1), 0), reinterpret_tensor(buf722, (512, ), (1, ), 0), reinterpret_tensor(buf717, (128, 512), (512, 1), 0), reinterpret_tensor(buf718, (128, ), (1, ), 0), reinterpret_tensor(buf711, (512, 128), (128, 1), 0), reinterpret_tensor(buf712, (512, ), (1, ), 0), reinterpret_tensor(buf707, (128, 512), (512, 1), 0), reinterpret_tensor(buf708, (128, ), (1, ), 0), reinterpret_tensor(buf701, (512, 128), (128, 1), 0), reinterpret_tensor(buf702, (512, ), (1, ), 0), reinterpret_tensor(buf697, (128, 512), (512, 1), 0), reinterpret_tensor(buf698, (128, ), (1, ), 0), reinterpret_tensor(buf691, (512, 128), (128, 1), 0), reinterpret_tensor(buf692, (512, ), (1, ), 0), reinterpret_tensor(buf685, (128, 512), (512, 1), 0), reinterpret_tensor(buf686, (128, ), (1, ), 0), reinterpret_tensor(buf679, (128, 512), (512, 1), 0), reinterpret_tensor(buf680, (128, ), (1, ), 0), reinterpret_tensor(buf673, (128, 128), (128, 1), 0), reinterpret_tensor(buf674, (128, ), (1, ), 0), reinterpret_tensor(buf670, (128, 128), (128, 1), 0), reinterpret_tensor(buf671, (128, ), (1, ), 0), reinterpret_tensor(buf667, (128, 512), (512, 1), 0), reinterpret_tensor(buf668, (128, ), (1, ), 0), reinterpret_tensor(buf660, (128, 128), (128, 1), 0), reinterpret_tensor(buf661, (128, ), (1, ), 0), reinterpret_tensor(buf654, (512, 128), (128, 1), 0), reinterpret_tensor(buf655, (512, ), (1, ), 0), reinterpret_tensor(buf650, (128, 512), (512, 1), 0), reinterpret_tensor(buf651, (128, ), (1, ), 0), reinterpret_tensor(buf644, (512, 128), (128, 1), 0), reinterpret_tensor(buf645, (512, ), (1, ), 0), reinterpret_tensor(buf640, (128, 512), (512, 1), 0), reinterpret_tensor(buf641, (128, ), (1, ), 0), reinterpret_tensor(buf634, (512, 128), (128, 1), 0), reinterpret_tensor(buf635, (512, ), (1, ), 0), reinterpret_tensor(buf630, (128, 512), (512, 1), 0), reinterpret_tensor(buf631, (128, ), (1, ), 0), reinterpret_tensor(buf624, (512, 128), (128, 1), 0), reinterpret_tensor(buf625, (512, ), (1, ), 0), reinterpret_tensor(buf620, (128, 512), (512, 1), 0), reinterpret_tensor(buf621, (128, ), (1, ), 0), reinterpret_tensor(buf614, (512, 128), (128, 1), 0), reinterpret_tensor(buf615, (512, ), (1, ), 0), reinterpret_tensor(buf608, (128, 512), (512, 1), 0), reinterpret_tensor(buf609, (128, ), (1, ), 0), reinterpret_tensor(buf602, (128, 512), (512, 1), 0), reinterpret_tensor(buf603, (128, ), (1, ), 0), reinterpret_tensor(buf596, (128, 128), (128, 1), 0), reinterpret_tensor(buf597, (128, ), (1, ), 0), reinterpret_tensor(buf593, (128, 128), (128, 1), 0), reinterpret_tensor(buf594, (128, ), (1, ), 0), reinterpret_tensor(buf590, (128, 512), (512, 1), 0), reinterpret_tensor(buf591, (128, ), (1, ), 0), reinterpret_tensor(buf583, (128, 128), (128, 1), 0), reinterpret_tensor(buf584, (128, ), (1, ), 0), reinterpret_tensor(buf577, (512, 128), (128, 1), 0), reinterpret_tensor(buf578, (512, ), (1, ), 0), reinterpret_tensor(buf573, (128, 512), (512, 1), 0), reinterpret_tensor(buf574, (128, ), (1, ), 0), reinterpret_tensor(buf567, (512, 128), (128, 1), 0), reinterpret_tensor(buf568, (512, ), (1, ), 0), reinterpret_tensor(buf563, (128, 512), (512, 1), 0), reinterpret_tensor(buf564, (128, ), (1, ), 0), reinterpret_tensor(buf557, (512, 128), (128, 1), 0), reinterpret_tensor(buf558, (512, ), (1, ), 0), reinterpret_tensor(buf553, (128, 512), (512, 1), 0), reinterpret_tensor(buf554, (128, ), (1, ), 0), reinterpret_tensor(buf547, (512, 128), (128, 1), 0), reinterpret_tensor(buf548, (512, ), (1, ), 0), reinterpret_tensor(buf543, (128, 512), (512, 1), 0), reinterpret_tensor(buf544, (128, ), (1, ), 0), reinterpret_tensor(buf537, (512, 128), (128, 1), 0), reinterpret_tensor(buf538, (512, ), (1, ), 0), reinterpret_tensor(buf531, (128, 512), (512, 1), 0), reinterpret_tensor(buf532, (128, ), (1, ), 0), reinterpret_tensor(buf525, (128, 512), (512, 1), 0), reinterpret_tensor(buf526, (128, ), (1, ), 0), reinterpret_tensor(buf519, (128, 128), (128, 1), 0), reinterpret_tensor(buf520, (128, ), (1, ), 0), reinterpret_tensor(buf516, (128, 128), (128, 1), 0), reinterpret_tensor(buf517, (128, ), (1, ), 0), reinterpret_tensor(buf513, (128, 512), (512, 1), 0), reinterpret_tensor(buf514, (128, ), (1, ), 0), reinterpret_tensor(buf506, (128, 128), (128, 1), 0), reinterpret_tensor(buf507, (128, ), (1, ), 0), reinterpret_tensor(buf500, (512, 128), (128, 1), 0), reinterpret_tensor(buf501, (512, ), (1, ), 0), reinterpret_tensor(buf496, (128, 512), (512, 1), 0), reinterpret_tensor(buf497, (128, ), (1, ), 0), reinterpret_tensor(buf490, (512, 128), (128, 1), 0), reinterpret_tensor(buf491, (512, ), (1, ), 0), reinterpret_tensor(buf486, (128, 512), (512, 1), 0), reinterpret_tensor(buf487, (128, ), (1, ), 0), reinterpret_tensor(buf480, (512, 128), (128, 1), 0), reinterpret_tensor(buf481, (512, ), (1, ), 0), reinterpret_tensor(buf476, (128, 512), (512, 1), 0), reinterpret_tensor(buf477, (128, ), (1, ), 0), reinterpret_tensor(buf470, (512, 128), (128, 1), 0), reinterpret_tensor(buf471, (512, ), (1, ), 0), reinterpret_tensor(buf466, (128, 512), (512, 1), 0), reinterpret_tensor(buf467, (128, ), (1, ), 0), reinterpret_tensor(buf460, (512, 128), (128, 1), 0), reinterpret_tensor(buf461, (512, ), (1, ), 0), reinterpret_tensor(buf454, (128, 512), (512, 1), 0), reinterpret_tensor(buf455, (128, ), (1, ), 0), reinterpret_tensor(buf448, (128, 512), (512, 1), 0), reinterpret_tensor(buf449, (128, ), (1, ), 0), reinterpret_tensor(buf442, (128, 128), (128, 1), 0), reinterpret_tensor(buf443, (128, ), (1, ), 0), reinterpret_tensor(buf439, (128, 128), (128, 1), 0), reinterpret_tensor(buf440, (128, ), (1, ), 0), reinterpret_tensor(buf436, (128, 512), (512, 1), 0), reinterpret_tensor(buf437, (128, ), (1, ), 0), reinterpret_tensor(buf429, (128, 128), (128, 1), 0), reinterpret_tensor(buf430, (128, ), (1, ), 0), reinterpret_tensor(buf423, (512, 128), (128, 1), 0), reinterpret_tensor(buf424, (512, ), (1, ), 0), reinterpret_tensor(buf419, (128, 512), (512, 1), 0), reinterpret_tensor(buf420, (128, ), (1, ), 0), reinterpret_tensor(buf413, (512, 128), (128, 1), 0), reinterpret_tensor(buf414, (512, ), (1, ), 0), reinterpret_tensor(buf409, (128, 512), (512, 1), 0), reinterpret_tensor(buf410, (128, ), (1, ), 0), reinterpret_tensor(buf403, (512, 128), (128, 1), 0), reinterpret_tensor(buf404, (512, ), (1, ), 0), reinterpret_tensor(buf399, (128, 512), (512, 1), 0), reinterpret_tensor(buf400, (128, ), (1, ), 0), reinterpret_tensor(buf393, (512, 128), (128, 1), 0), reinterpret_tensor(buf394, (512, ), (1, ), 0), reinterpret_tensor(buf389, (128, 512), (512, 1), 0), reinterpret_tensor(buf390, (128, ), (1, ), 0), reinterpret_tensor(buf383, (512, 128), (128, 1), 0), reinterpret_tensor(buf384, (512, ), (1, ), 0), reinterpret_tensor(buf377, (128, 512), (512, 1), 0), reinterpret_tensor(buf378, (128, ), (1, ), 0), reinterpret_tensor(buf371, (128, 512), (512, 1), 0), reinterpret_tensor(buf372, (128, ), (1, ), 0), reinterpret_tensor(buf365, (128, 128), (128, 1), 0), reinterpret_tensor(buf366, (128, ), (1, ), 0), reinterpret_tensor(buf362, (128, 128), (128, 1), 0), reinterpret_tensor(buf363, (128, ), (1, ), 0), reinterpret_tensor(buf359, (128, 512), (512, 1), 0), reinterpret_tensor(buf360, (128, ), (1, ), 0), reinterpret_tensor(buf352, (128, 128), (128, 1), 0), reinterpret_tensor(buf353, (128, ), (1, ), 0), reinterpret_tensor(buf346, (512, 128), (128, 1), 0), reinterpret_tensor(buf347, (512, ), (1, ), 0), reinterpret_tensor(buf342, (128, 512), (512, 1), 0), reinterpret_tensor(buf343, (128, ), (1, ), 0), reinterpret_tensor(buf336, (512, 128), (128, 1), 0), reinterpret_tensor(buf337, (512, ), (1, ), 0), reinterpret_tensor(buf332, (128, 512), (512, 1), 0), reinterpret_tensor(buf333, (128, ), (1, ), 0), reinterpret_tensor(buf326, (512, 128), (128, 1), 0), reinterpret_tensor(buf327, (512, ), (1, ), 0), reinterpret_tensor(buf322, (128, 512), (512, 1), 0), reinterpret_tensor(buf323, (128, ), (1, ), 0), reinterpret_tensor(buf316, (512, 128), (128, 1), 0), reinterpret_tensor(buf317, (512, ), (1, ), 0), reinterpret_tensor(buf312, (128, 512), (512, 1), 0), reinterpret_tensor(buf313, (128, ), (1, ), 0), reinterpret_tensor(buf306, (512, 128), (128, 1), 0), reinterpret_tensor(buf307, (512, ), (1, ), 0), reinterpret_tensor(buf300, (128, 512), (512, 1), 0), reinterpret_tensor(buf301, (128, ), (1, ), 0), reinterpret_tensor(buf294, (128, 512), (512, 1), 0), reinterpret_tensor(buf295, (128, ), (1, ), 0), reinterpret_tensor(buf288, (128, 128), (128, 1), 0), reinterpret_tensor(buf289, (128, ), (1, ), 0), reinterpret_tensor(buf285, (128, 128), (128, 1), 0), reinterpret_tensor(buf286, (128, ), (1, ), 0), reinterpret_tensor(buf282, (128, 512), (512, 1), 0), reinterpret_tensor(buf283, (128, ), (1, ), 0), reinterpret_tensor(buf275, (128, 128), (128, 1), 0), reinterpret_tensor(buf276, (128, ), (1, ), 0), reinterpret_tensor(buf269, (512, 128), (128, 1), 0), reinterpret_tensor(buf270, (512, ), (1, ), 0), reinterpret_tensor(buf265, (128, 512), (512, 1), 0), reinterpret_tensor(buf266, (128, ), (1, ), 0), reinterpret_tensor(buf259, (512, 128), (128, 1), 0), reinterpret_tensor(buf260, (512, ), (1, ), 0), reinterpret_tensor(buf255, (128, 512), (512, 1), 0), reinterpret_tensor(buf256, (128, ), (1, ), 0), reinterpret_tensor(buf249, (512, 128), (128, 1), 0), reinterpret_tensor(buf250, (512, ), (1, ), 0), reinterpret_tensor(buf245, (128, 512), (512, 1), 0), reinterpret_tensor(buf246, (128, ), (1, ), 0), reinterpret_tensor(buf239, (512, 128), (128, 1), 0), reinterpret_tensor(buf240, (512, ), (1, ), 0), reinterpret_tensor(buf235, (128, 512), (512, 1), 0), reinterpret_tensor(buf236, (128, ), (1, ), 0), reinterpret_tensor(buf229, (512, 128), (128, 1), 0), reinterpret_tensor(buf230, (512, ), (1, ), 0), reinterpret_tensor(buf223, (128, 512), (512, 1), 0), reinterpret_tensor(buf224, (128, ), (1, ), 0), reinterpret_tensor(buf217, (128, 512), (512, 1), 0), reinterpret_tensor(buf218, (128, ), (1, ), 0), reinterpret_tensor(buf211, (128, 128), (128, 1), 0), reinterpret_tensor(buf212, (128, ), (1, ), 0), reinterpret_tensor(buf208, (128, 128), (128, 1), 0), reinterpret_tensor(buf209, (128, ), (1, ), 0), reinterpret_tensor(buf205, (128, 512), (512, 1), 0), reinterpret_tensor(buf206, (128, ), (1, ), 0), reinterpret_tensor(buf198, (128, 128), (128, 1), 0), reinterpret_tensor(buf199, (128, ), (1, ), 0), reinterpret_tensor(buf192, (512, 128), (128, 1), 0), reinterpret_tensor(buf193, (512, ), (1, ), 0), reinterpret_tensor(buf188, (128, 512), (512, 1), 0), reinterpret_tensor(buf189, (128, ), (1, ), 0), reinterpret_tensor(buf182, (512, 128), (128, 1), 0), reinterpret_tensor(buf183, (512, ), (1, ), 0), reinterpret_tensor(buf178, (128, 512), (512, 1), 0), reinterpret_tensor(buf179, (128, ), (1, ), 0), reinterpret_tensor(buf172, (512, 128), (128, 1), 0), reinterpret_tensor(buf173, (512, ), (1, ), 0), reinterpret_tensor(buf168, (128, 512), (512, 1), 0), reinterpret_tensor(buf169, (128, ), (1, ), 0), reinterpret_tensor(buf162, (512, 128), (128, 1), 0), reinterpret_tensor(buf163, (512, ), (1, ), 0), reinterpret_tensor(buf158, (128, 512), (512, 1), 0), reinterpret_tensor(buf159, (128, ), (1, ), 0), reinterpret_tensor(buf152, (512, 128), (128, 1), 0), reinterpret_tensor(buf153, (512, ), (1, ), 0), reinterpret_tensor(buf146, (128, 512), (512, 1), 0), reinterpret_tensor(buf147, (128, ), (1, ), 0), reinterpret_tensor(buf140, (128, 512), (512, 1), 0), reinterpret_tensor(buf141, (128, ), (1, ), 0), reinterpret_tensor(buf134, (128, 128), (128, 1), 0), reinterpret_tensor(buf135, (128, ), (1, ), 0), reinterpret_tensor(buf131, (128, 128), (128, 1), 0), reinterpret_tensor(buf132, (128, ), (1, ), 0), reinterpret_tensor(buf128, (128, 512), (512, 1), 0), reinterpret_tensor(buf129, (128, ), (1, ), 0), reinterpret_tensor(buf121, (128, 128), (128, 1), 0), reinterpret_tensor(buf122, (128, ), (1, ), 0), reinterpret_tensor(buf115, (512, 128), (128, 1), 0), reinterpret_tensor(buf116, (512, ), (1, ), 0), reinterpret_tensor(buf111, (128, 512), (512, 1), 0), reinterpret_tensor(buf112, (128, ), (1, ), 0), reinterpret_tensor(buf105, (512, 128), (128, 1), 0), reinterpret_tensor(buf106, (512, ), (1, ), 0), reinterpret_tensor(buf101, (128, 512), (512, 1), 0), reinterpret_tensor(buf102, (128, ), (1, ), 0), reinterpret_tensor(buf95, (512, 128), (128, 1), 0), reinterpret_tensor(buf96, (512, ), (1, ), 0), reinterpret_tensor(buf91, (128, 512), (512, 1), 0), reinterpret_tensor(buf92, (128, ), (1, ), 0), reinterpret_tensor(buf85, (512, 128), (128, 1), 0), reinterpret_tensor(buf86, (512, ), (1, ), 0), reinterpret_tensor(buf81, (128, 512), (512, 1), 0), reinterpret_tensor(buf82, (128, ), (1, ), 0), reinterpret_tensor(buf75, (512, 128), (128, 1), 0), reinterpret_tensor(buf76, (512, ), (1, ), 0), reinterpret_tensor(buf69, (2, 512), (512, 1), 0), reinterpret_tensor(buf70, (2, ), (1, ), 0), None, None, None, None, )


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
    primals_1115 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
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
    sub_26 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    ne = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.bool)
    sub_28 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    ne_3 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.bool)
    ne_6 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_4 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_8 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_6 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    permute_482 = rand_strided((2, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_486 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_490 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_494 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_498 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_1 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_502 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_506 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_2 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_510 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_514 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_3 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_518 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_522 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_535 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_539 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_543 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_547 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_551 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_555 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_559 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_4 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_563 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_567 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_5 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_571 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_575 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_6 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_579 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_583 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_7 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_587 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_591 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_604 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_608 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_612 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_616 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_620 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_624 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_628 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_8 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_632 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_636 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_9 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_640 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_644 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_10 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_648 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_652 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_11 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_656 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_660 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_673 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_677 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_681 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_685 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_689 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_693 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_697 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_12 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_701 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_705 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_13 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_709 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_713 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_14 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_717 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_721 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_15 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_725 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_729 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_742 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_746 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_750 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_754 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_758 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_762 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_766 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_16 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_770 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_774 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_17 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_778 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_782 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_18 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_786 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_790 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_19 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_794 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_798 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_811 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_815 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_819 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_823 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_827 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_831 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_835 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_20 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_839 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_843 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_21 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_847 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_851 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_22 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_855 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_859 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_23 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_863 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_867 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_880 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_884 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_888 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_892 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_896 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_900 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_904 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_24 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_908 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_912 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_25 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_916 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_920 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_26 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_924 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_928 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_27 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_932 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_936 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_949 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_953 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_957 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_961 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_965 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_969 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_973 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_28 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_977 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_981 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_29 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_985 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_989 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_30 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_993 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_997 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_31 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1001 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1005 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1018 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1022 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1026 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1030 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1034 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1038 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1042 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_32 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1046 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1050 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_33 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1054 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1058 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_34 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1062 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1066 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_35 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1070 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1074 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1087 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1091 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1095 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1099 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1103 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1107 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1111 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_36 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1115 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1119 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_37 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1123 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1127 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_38 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1131 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1135 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_39 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1139 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1143 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1156 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1160 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1164 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1168 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1172 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1176 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1180 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_40 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1184 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1188 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_41 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1192 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1196 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_42 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1200 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1204 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_43 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1208 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1212 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1225 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1229 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1233 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1237 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1241 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1245 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1249 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_44 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1253 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1257 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_45 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1261 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1265 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_46 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1269 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1273 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_47 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1277 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1281 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1294 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1298 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1302 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1306 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1310 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1314 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1318 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_48 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1322 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1326 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_49 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1330 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1334 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_50 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1338 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1342 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_51 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1346 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1350 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1363 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1367 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1371 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1375 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1379 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1383 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1387 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_52 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1391 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1395 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_53 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1399 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1403 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_54 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1407 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1411 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_55 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1415 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1419 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1432 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1436 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1440 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1444 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1448 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1452 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1456 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_56 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1460 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1464 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_57 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1468 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1472 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_58 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1476 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1480 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_59 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1484 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1488 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1501 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1505 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1509 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1513 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1517 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1521 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1525 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_60 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1529 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1533 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_61 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1537 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1541 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_62 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1545 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1549 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_63 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1553 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1557 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1570 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1574 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1578 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1582 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1586 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1590 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1594 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_64 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1598 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1602 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_65 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1606 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1610 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_66 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1614 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1618 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_67 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1622 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1626 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1639 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1643 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1647 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1651 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1655 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1659 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1663 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_68 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1667 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1671 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_69 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1675 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1679 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_70 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1683 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1687 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_71 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1691 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1695 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1708 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1712 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1716 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1720 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1724 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1728 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1732 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_72 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1736 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1740 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_73 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1744 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1748 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_74 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1752 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1756 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_75 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1760 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1764 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1777 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1781 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1785 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1789 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1793 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1797 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1801 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_76 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1805 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1809 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_77 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1813 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1817 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_78 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1821 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1825 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_79 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1829 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1833 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1846 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1850 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1854 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1858 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1862 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1866 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1870 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_80 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1874 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1878 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_81 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1882 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1886 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_82 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1890 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1894 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_83 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1898 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1902 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1915 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1919 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1923 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1927 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1931 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1935 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1939 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_84 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1943 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1947 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_85 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1951 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1955 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_86 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1959 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1963 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_87 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_1967 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1971 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1984 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_1988 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1992 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_1996 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_2000 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_2004 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2008 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_88 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_2012 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2016 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_89 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_2020 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2024 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_90 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_2028 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2032 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_91 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_2036 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2040 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2053 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_2057 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2061 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2065 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_2069 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_2073 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2077 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_92 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_2081 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2085 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_93 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_2089 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2093 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_94 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_2097 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2101 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    le_95 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    permute_2105 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2109 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2122 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_2126 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2130 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_2134 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_2138 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_2142 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_21, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_36, primals_37, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_49, primals_50, primals_51, primals_52, primals_53, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_65, primals_66, primals_67, primals_68, primals_69, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_81, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_97, primals_98, primals_99, primals_100, primals_101, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_113, primals_114, primals_115, primals_116, primals_117, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_129, primals_130, primals_131, primals_132, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_145, primals_146, primals_147, primals_148, primals_149, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_161, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_177, primals_178, primals_179, primals_180, primals_181, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_193, primals_194, primals_195, primals_196, primals_197, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_209, primals_210, primals_211, primals_212, primals_213, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_225, primals_226, primals_227, primals_228, primals_229, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_241, primals_242, primals_243, primals_244, primals_245, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_257, primals_258, primals_259, primals_260, primals_261, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_273, primals_274, primals_275, primals_276, primals_277, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_289, primals_290, primals_291, primals_292, primals_293, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_305, primals_306, primals_307, primals_308, primals_309, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_321, primals_322, primals_323, primals_324, primals_325, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_337, primals_338, primals_339, primals_340, primals_341, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_353, primals_354, primals_355, primals_356, primals_357, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_369, primals_370, primals_371, primals_372, primals_373, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_385, primals_1115, full_default, slice_4, view, add_1, view_2, addmm_1, addmm_2, view_6, clone_default_69, clone_default_70, clone_default_71, getitem_212, getitem_213, getitem_214, alias_default_47, view_22, addmm_6, view_24, view_26, addmm_8, view_28, view_30, addmm_10, view_32, view_34, addmm_12, view_36, view_38, addmm_14, view_40, add_16, view_42, addmm_16, addmm_17, view_46, clone_default_66, clone_default_67, clone_default_68, getitem_205, getitem_206, getitem_207, alias_default_45, view_62, addmm_21, view_64, view_66, addmm_23, view_68, view_70, addmm_25, view_72, view_74, addmm_27, view_76, view_78, addmm_29, view_80, addmm_30, view_82, addmm_31, addmm_32, view_86, clone_default_63, clone_default_64, clone_default_65, getitem_198, getitem_199, getitem_200, alias_default_43, view_102, addmm_36, view_104, view_106, addmm_38, view_108, view_110, addmm_40, view_112, view_114, addmm_42, view_116, view_118, addmm_44, view_120, addmm_45, view_122, addmm_46, addmm_47, view_126, clone_default_60, clone_default_61, clone_default_62, getitem_191, getitem_192, getitem_193, alias_default_41, view_142, addmm_51, view_144, view_146, addmm_53, view_148, view_150, addmm_55, view_152, view_154, addmm_57, view_156, view_158, addmm_59, view_160, addmm_60, view_162, addmm_61, addmm_62, view_166, clone_default_57, clone_default_58, clone_default_59, getitem_184, getitem_185, getitem_186, alias_default_39, view_182, addmm_66, view_184, view_186, addmm_68, view_188, view_190, addmm_70, view_192, view_194, addmm_72, view_196, view_198, addmm_74, view_200, addmm_75, view_202, addmm_76, addmm_77, view_206, clone_default_54, clone_default_55, clone_default_56, getitem_177, getitem_178, getitem_179, alias_default_37, view_222, addmm_81, view_224, view_226, addmm_83, view_228, view_230, addmm_85, view_232, view_234, addmm_87, view_236, view_238, addmm_89, view_240, addmm_90, view_242, addmm_91, addmm_92, view_246, clone_default_51, clone_default_52, clone_default_53, getitem_170, getitem_171, getitem_172, alias_default_35, view_262, addmm_96, view_264, view_266, addmm_98, view_268, view_270, addmm_100, view_272, view_274, addmm_102, view_276, view_278, addmm_104, view_280, addmm_105, view_282, addmm_106, addmm_107, view_286, clone_default_48, clone_default_49, clone_default_50, getitem_163, getitem_164, getitem_165, alias_default_33, view_302, addmm_111, view_304, view_306, addmm_113, view_308, view_310, addmm_115, view_312, view_314, addmm_117, view_316, view_318, addmm_119, view_320, addmm_120, view_322, addmm_121, addmm_122, view_326, clone_default_45, clone_default_46, clone_default_47, getitem_156, getitem_157, getitem_158, alias_default_31, view_342, addmm_126, view_344, view_346, addmm_128, view_348, view_350, addmm_130, view_352, view_354, addmm_132, view_356, view_358, addmm_134, view_360, addmm_135, view_362, addmm_136, addmm_137, view_366, clone_default_42, clone_default_43, clone_default_44, getitem_149, getitem_150, getitem_151, alias_default_29, view_382, addmm_141, view_384, view_386, addmm_143, view_388, view_390, addmm_145, view_392, view_394, addmm_147, view_396, view_398, addmm_149, view_400, addmm_150, view_402, addmm_151, addmm_152, view_406, clone_default_39, clone_default_40, clone_default_41, getitem_142, getitem_143, getitem_144, alias_default_27, view_422, addmm_156, view_424, view_426, addmm_158, view_428, view_430, addmm_160, view_432, view_434, addmm_162, view_436, view_438, addmm_164, view_440, addmm_165, view_442, addmm_166, addmm_167, view_446, clone_default_36, clone_default_37, clone_default_38, getitem_135, getitem_136, getitem_137, alias_default_25, view_462, addmm_171, view_464, view_466, addmm_173, view_468, view_470, addmm_175, view_472, view_474, addmm_177, view_476, view_478, addmm_179, view_480, addmm_180, view_482, addmm_181, addmm_182, view_486, clone_default_33, clone_default_34, clone_default_35, getitem_128, getitem_129, getitem_130, alias_default_23, view_502, addmm_186, view_504, view_506, addmm_188, view_508, view_510, addmm_190, view_512, view_514, addmm_192, view_516, view_518, addmm_194, view_520, addmm_195, view_522, addmm_196, addmm_197, view_526, clone_default_30, clone_default_31, clone_default_32, getitem_121, getitem_122, getitem_123, alias_default_21, view_542, addmm_201, view_544, view_546, addmm_203, view_548, view_550, addmm_205, view_552, view_554, addmm_207, view_556, view_558, addmm_209, view_560, addmm_210, view_562, addmm_211, addmm_212, view_566, clone_default_27, clone_default_28, clone_default_29, getitem_114, getitem_115, getitem_116, alias_default_19, view_582, addmm_216, view_584, view_586, addmm_218, view_588, view_590, addmm_220, view_592, view_594, addmm_222, view_596, view_598, addmm_224, view_600, addmm_225, view_602, addmm_226, addmm_227, view_606, clone_default_24, clone_default_25, clone_default_26, getitem_107, getitem_108, getitem_109, alias_default_17, view_622, addmm_231, view_624, view_626, addmm_233, view_628, view_630, addmm_235, view_632, view_634, addmm_237, view_636, view_638, addmm_239, view_640, addmm_240, view_642, addmm_241, addmm_242, view_646, clone_default_21, clone_default_22, clone_default_23, getitem_100, getitem_101, getitem_102, alias_default_15, view_662, addmm_246, view_664, view_666, addmm_248, view_668, view_670, addmm_250, view_672, view_674, addmm_252, view_676, view_678, addmm_254, view_680, addmm_255, view_682, addmm_256, addmm_257, view_686, clone_default_18, clone_default_19, clone_default_20, getitem_93, getitem_94, getitem_95, alias_default_13, view_702, addmm_261, view_704, view_706, addmm_263, view_708, view_710, addmm_265, view_712, view_714, addmm_267, view_716, view_718, addmm_269, view_720, addmm_270, view_722, addmm_271, addmm_272, view_726, clone_default_15, clone_default_16, clone_default_17, getitem_86, getitem_87, getitem_88, alias_default_11, view_742, addmm_276, view_744, view_746, addmm_278, view_748, view_750, addmm_280, view_752, view_754, addmm_282, view_756, view_758, addmm_284, view_760, addmm_285, view_762, addmm_286, addmm_287, view_766, clone_default_12, clone_default_13, clone_default_14, getitem_79, getitem_80, getitem_81, alias_default_9, view_782, addmm_291, view_784, view_786, addmm_293, view_788, view_790, addmm_295, view_792, view_794, addmm_297, view_796, view_798, addmm_299, view_800, addmm_300, view_802, addmm_301, addmm_302, view_806, clone_default_9, clone_default_10, clone_default_11, getitem_72, getitem_73, getitem_74, alias_default_7, view_822, addmm_306, view_824, view_826, addmm_308, view_828, view_830, addmm_310, view_832, view_834, addmm_312, view_836, view_838, addmm_314, view_840, addmm_315, view_842, addmm_316, addmm_317, view_846, clone_default_6, clone_default_7, clone_default_8, getitem_65, getitem_66, getitem_67, alias_default_5, view_862, addmm_321, view_864, view_866, addmm_323, view_868, view_870, addmm_325, view_872, view_874, addmm_327, view_876, view_878, addmm_329, view_880, addmm_330, view_882, addmm_331, addmm_332, view_886, clone_default_3, clone_default_4, clone_default_5, getitem_58, getitem_59, getitem_60, alias_default_3, view_902, addmm_336, view_904, view_906, addmm_338, view_908, view_910, addmm_340, view_912, view_914, addmm_342, view_916, view_918, addmm_344, view_920, addmm_345, view_922, addmm_346, addmm_347, view_926, clone_default, clone_default_1, clone_default_2, getitem_51, getitem_52, getitem_53, alias_default_1, view_942, addmm_351, view_944, view_946, addmm_353, view_948, view_950, addmm_355, view_952, view_954, addmm_357, view_956, view_958, addmm_359, view_960, addmm_360, view_962, sub_26, ne, sub_28, ne_3, ne_6, where_4, ne_8, where_6, permute_482, permute_486, permute_490, le, permute_494, permute_498, le_1, permute_502, permute_506, le_2, permute_510, permute_514, le_3, permute_518, permute_522, permute_535, permute_539, permute_543, permute_547, permute_551, permute_555, permute_559, le_4, permute_563, permute_567, le_5, permute_571, permute_575, le_6, permute_579, permute_583, le_7, permute_587, permute_591, permute_604, permute_608, permute_612, permute_616, permute_620, permute_624, permute_628, le_8, permute_632, permute_636, le_9, permute_640, permute_644, le_10, permute_648, permute_652, le_11, permute_656, permute_660, permute_673, permute_677, permute_681, permute_685, permute_689, permute_693, permute_697, le_12, permute_701, permute_705, le_13, permute_709, permute_713, le_14, permute_717, permute_721, le_15, permute_725, permute_729, permute_742, permute_746, permute_750, permute_754, permute_758, permute_762, permute_766, le_16, permute_770, permute_774, le_17, permute_778, permute_782, le_18, permute_786, permute_790, le_19, permute_794, permute_798, permute_811, permute_815, permute_819, permute_823, permute_827, permute_831, permute_835, le_20, permute_839, permute_843, le_21, permute_847, permute_851, le_22, permute_855, permute_859, le_23, permute_863, permute_867, permute_880, permute_884, permute_888, permute_892, permute_896, permute_900, permute_904, le_24, permute_908, permute_912, le_25, permute_916, permute_920, le_26, permute_924, permute_928, le_27, permute_932, permute_936, permute_949, permute_953, permute_957, permute_961, permute_965, permute_969, permute_973, le_28, permute_977, permute_981, le_29, permute_985, permute_989, le_30, permute_993, permute_997, le_31, permute_1001, permute_1005, permute_1018, permute_1022, permute_1026, permute_1030, permute_1034, permute_1038, permute_1042, le_32, permute_1046, permute_1050, le_33, permute_1054, permute_1058, le_34, permute_1062, permute_1066, le_35, permute_1070, permute_1074, permute_1087, permute_1091, permute_1095, permute_1099, permute_1103, permute_1107, permute_1111, le_36, permute_1115, permute_1119, le_37, permute_1123, permute_1127, le_38, permute_1131, permute_1135, le_39, permute_1139, permute_1143, permute_1156, permute_1160, permute_1164, permute_1168, permute_1172, permute_1176, permute_1180, le_40, permute_1184, permute_1188, le_41, permute_1192, permute_1196, le_42, permute_1200, permute_1204, le_43, permute_1208, permute_1212, permute_1225, permute_1229, permute_1233, permute_1237, permute_1241, permute_1245, permute_1249, le_44, permute_1253, permute_1257, le_45, permute_1261, permute_1265, le_46, permute_1269, permute_1273, le_47, permute_1277, permute_1281, permute_1294, permute_1298, permute_1302, permute_1306, permute_1310, permute_1314, permute_1318, le_48, permute_1322, permute_1326, le_49, permute_1330, permute_1334, le_50, permute_1338, permute_1342, le_51, permute_1346, permute_1350, permute_1363, permute_1367, permute_1371, permute_1375, permute_1379, permute_1383, permute_1387, le_52, permute_1391, permute_1395, le_53, permute_1399, permute_1403, le_54, permute_1407, permute_1411, le_55, permute_1415, permute_1419, permute_1432, permute_1436, permute_1440, permute_1444, permute_1448, permute_1452, permute_1456, le_56, permute_1460, permute_1464, le_57, permute_1468, permute_1472, le_58, permute_1476, permute_1480, le_59, permute_1484, permute_1488, permute_1501, permute_1505, permute_1509, permute_1513, permute_1517, permute_1521, permute_1525, le_60, permute_1529, permute_1533, le_61, permute_1537, permute_1541, le_62, permute_1545, permute_1549, le_63, permute_1553, permute_1557, permute_1570, permute_1574, permute_1578, permute_1582, permute_1586, permute_1590, permute_1594, le_64, permute_1598, permute_1602, le_65, permute_1606, permute_1610, le_66, permute_1614, permute_1618, le_67, permute_1622, permute_1626, permute_1639, permute_1643, permute_1647, permute_1651, permute_1655, permute_1659, permute_1663, le_68, permute_1667, permute_1671, le_69, permute_1675, permute_1679, le_70, permute_1683, permute_1687, le_71, permute_1691, permute_1695, permute_1708, permute_1712, permute_1716, permute_1720, permute_1724, permute_1728, permute_1732, le_72, permute_1736, permute_1740, le_73, permute_1744, permute_1748, le_74, permute_1752, permute_1756, le_75, permute_1760, permute_1764, permute_1777, permute_1781, permute_1785, permute_1789, permute_1793, permute_1797, permute_1801, le_76, permute_1805, permute_1809, le_77, permute_1813, permute_1817, le_78, permute_1821, permute_1825, le_79, permute_1829, permute_1833, permute_1846, permute_1850, permute_1854, permute_1858, permute_1862, permute_1866, permute_1870, le_80, permute_1874, permute_1878, le_81, permute_1882, permute_1886, le_82, permute_1890, permute_1894, le_83, permute_1898, permute_1902, permute_1915, permute_1919, permute_1923, permute_1927, permute_1931, permute_1935, permute_1939, le_84, permute_1943, permute_1947, le_85, permute_1951, permute_1955, le_86, permute_1959, permute_1963, le_87, permute_1967, permute_1971, permute_1984, permute_1988, permute_1992, permute_1996, permute_2000, permute_2004, permute_2008, le_88, permute_2012, permute_2016, le_89, permute_2020, permute_2024, le_90, permute_2028, permute_2032, le_91, permute_2036, permute_2040, permute_2053, permute_2057, permute_2061, permute_2065, permute_2069, permute_2073, permute_2077, le_92, permute_2081, permute_2085, le_93, permute_2089, permute_2093, le_94, permute_2097, permute_2101, le_95, permute_2105, permute_2109, permute_2122, permute_2126, permute_2130, permute_2134, permute_2138, permute_2142, tangents_1, tangents_2, tangents_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MobileBertForQuestionAnswering', benchmark_compiled_module)
