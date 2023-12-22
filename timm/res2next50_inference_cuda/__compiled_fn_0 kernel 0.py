
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


# kernel path: /tmp/torchinductor_youkaichao/63/c63tcifdfouzgxewkdgeawixxyvaylaegdnohhvu7jymv2qzuy4c.py
# Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_1 => add_1, mul_1, mul_2, sub
# x_2 => relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/sv/csvbla5mz5jgtaoekyjsm4ld7uiquhja3wn5hcce4sp6hhuvfsh5.py
# Source Nodes: [shortcut, x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.max_pool2d_with_indices, aten.relu]
# shortcut => max_pool2d_with_indices
# x_1 => add_1, mul_1, mul_2, sub
# x_2 => relu
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x3 = (xindex // 56)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-113) + (2*x0) + (224*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-112) + (2*x0) + (224*x3)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-111) + (2*x0) + (224*x3)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + (2*x0) + (224*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + ((2*x0) + (224*x3)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + (2*x0) + (224*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (111 + (2*x0) + (224*x3)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (112 + (2*x0) + (224*x3)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (113 + (2*x0) + (224*x3)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tl.store(out_ptr0 + (x4), tmp69, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pl/cplud5se2ybdnaccoaiw4sz5hutpl6demhqf2s5o2z4fz25jcqhh.py
# Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_1 => add_3, mul_4, mul_5, sub_1
# out_2 => relu_1
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zo/czojfm2gquxzm7a744lf755vkxetxgstigerfadyyqohw6g4x5wv.py
# Source Nodes: [getattr_l__mod___layer1___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer1___0___pool => avg_pool2d
triton_poi_fused_avg_pool2d_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x3 = (xindex // 100352)
    x6 = xindex % 100352
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (300999 + x6 + (401408*x3)), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (301000 + x6 + (401408*x3)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (301001 + x6 + (401408*x3)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (301055 + x6 + (401408*x3)), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (301056 + x6 + (401408*x3)), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (301057 + x6 + (401408*x3)), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x1
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (301111 + x6 + (401408*x3)), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (301112 + x6 + (401408*x3)), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (301113 + x6 + (401408*x3)), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 57, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x6 + (401408*x3)), tmp145, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lu/cluelrwu3g5hhlwzjanwlrxs3bletkyvh4ewhfuwosajn5uzrfzx.py
# Source Nodes: [sp_2, sp_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# sp_2 => add_5, mul_7, mul_8, sub_2
# sp_3 => relu_2
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 32
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x4 + (401408*x2)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xf/cxfdegbumlpaml5td4ktelhd2mpin6xzvkhism5qvlby7qsk3y5f.py
# Source Nodes: [out_5, out_6, shortcut_1, shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_5 => add_11, mul_16, mul_17, sub_5
# out_6 => add_14
# shortcut_1 => add_13, mul_19, mul_20, sub_6
# shortcut_2 => relu_5
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(in_out_ptr0 + (x3), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/73/c73422kahi3nspofnitrwuc6ytyj5szdrxpvbiay4nkgds6ywefp.py
# Source Nodes: [sp_15, sp_16, sp_17, sp_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
# sp_15 => add_18, mul_25, mul_26, sub_8
# sp_16 => relu_7
# sp_17 => add_19
# sp_18 => convolution_9
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 32
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (100352 + x4 + (401408*x2)), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x4 + (401408*x2)), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7f/c7fweziuk6qbgmlq3idod3vk2asep2oy5bzsoqjdq3qijvpbte5v.py
# Source Nodes: [sp_19, sp_20, sp_21, sp_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
# sp_19 => add_21, mul_28, mul_29, sub_9
# sp_20 => relu_8
# sp_21 => add_22
# sp_22 => convolution_10
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 32
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (200704 + x4 + (401408*x2)), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x4 + (401408*x2)), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4h/c4hn5e3kumdipx6vmowf4hcd4y2bqmzgktnvuf6asnkjslsjehm4.py
# Source Nodes: [cat_30], Original ATen: [aten.cat]
# cat_30 => cat_1
triton_poi_fused_cat_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 100352
    x1 = (xindex // 100352)
    tmp0 = tl.load(in_ptr0 + (301056 + x0 + (401408*x1)), None)
    tl.store(out_ptr0 + (x0 + (401408*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/23/c23ppngqkmmgrxjz7bqqgbbiy4gwfrmmg4jydaspg7u4g3jvuy6x.py
# Source Nodes: [out_13, out_14, shortcut_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_13 => add_26, mul_34, mul_35, sub_11
# out_14 => add_27
# shortcut_3 => relu_10
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l4/cl44z3fvagcfmqwyxyullgg3xprg6kij73mdsfhiknis24ayouym.py
# Source Nodes: [out_25, out_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_25 => add_42, mul_52, mul_53, sub_17
# out_26 => relu_16
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qe/cqejudmejaj5fq6ot2ey6a5ozvavwyvaa3ooegluvmmlkrokprmm.py
# Source Nodes: [getattr_l__mod___layer2___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer2___0___pool => avg_pool2d_1
triton_poi_fused_avg_pool2d_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    x3 = (xindex // 50176)
    x6 = (xindex // 28) % 1792
    x7 = xindex % 50176
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (602055 + (2*x0) + (112*x6) + (802816*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (602056 + (2*x0) + (112*x6) + (802816*x3)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (602057 + (2*x0) + (112*x6) + (802816*x3)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (602111 + (2*x0) + (112*x6) + (802816*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (602112 + (2*x0) + (112*x6) + (802816*x3)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (602113 + (2*x0) + (112*x6) + (802816*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (602167 + (2*x0) + (112*x6) + (802816*x3)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (602168 + (2*x0) + (112*x6) + (802816*x3)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (602169 + (2*x0) + (112*x6) + (802816*x3)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 57, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x7 + (200704*x3)), tmp145, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gb/cgb3stshmij3y754tpkwd3j7ogeacx7hw5i4tfqz25wexz3utwiq.py
# Source Nodes: [sp_41, sp_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# sp_41 => add_44, mul_55, mul_56, sub_18
# sp_42 => relu_17
triton_poi_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 64
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x4 + (200704*x2)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kw/ckwbv5swyrokacq5adthhsxwlmgessbzdvr3d6zk5e4lhegi6yzg.py
# Source Nodes: [out_29, out_30, shortcut_5, shortcut_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_29 => add_50, mul_64, mul_65, sub_21
# out_30 => add_53
# shortcut_5 => add_52, mul_67, mul_68, sub_22
# shortcut_6 => relu_20
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(in_out_ptr0 + (x3), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q7/cq7iuwinqjcvneanadz4sr5xi5nn2j5pabd4wgejuop7ymy4xi6v.py
# Source Nodes: [out_33, out_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_33 => add_55, mul_70, mul_71, sub_23
# out_34 => relu_21
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jg/cjg66yf54snkozfoxdin57jbnhajt376yclkktec4ckm4hma5m4m.py
# Source Nodes: [sp_54, sp_55, sp_56, sp_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
# sp_54 => add_57, mul_73, mul_74, sub_24
# sp_55 => relu_22
# sp_56 => add_58
# sp_57 => convolution_25
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 64
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (50176 + x4 + (200704*x2)), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x4 + (200704*x2)), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wp/cwphse7ewh7sa3c37rz4snkd3tifgj26bsocirlrj6jglllyrbdw.py
# Source Nodes: [sp_58, sp_59, sp_60, sp_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
# sp_58 => add_60, mul_76, mul_77, sub_25
# sp_59 => relu_23
# sp_60 => add_61
# sp_61 => convolution_26
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 64
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (100352 + x4 + (200704*x2)), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x4 + (200704*x2)), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tr/ctrfxjredfco3x77ssvcprvquapzn6ua3fkuji7yplq5iu6r7yyr.py
# Source Nodes: [cat_27], Original ATen: [aten.cat]
# cat_27 => cat_4
triton_poi_fused_cat_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 50176
    x1 = (xindex // 50176)
    tmp0 = tl.load(in_ptr0 + (150528 + x0 + (200704*x1)), None)
    tl.store(out_ptr0 + (x0 + (200704*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yb/cybmwdgsen3epvawqruaxwoxqtld6gaxqojsusg2c76mxz2cwiwx.py
# Source Nodes: [out_37, out_38, shortcut_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_37 => add_65, mul_82, mul_83, sub_27
# out_38 => add_66
# shortcut_7 => relu_25
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jq/cjqxm7ygo2hh5v2kifgudkrtrzh5h52f25jmbgrtberfgtyvnfet.py
# Source Nodes: [out_57, out_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_57 => add_94, mul_115, mul_116, sub_38
# out_58 => relu_36
triton_poi_fused__native_batch_norm_legit_no_training_relu_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4l/c4l7jzszmv64gl7iex2zb7akcauhh4gi25iqx2cqzdaf62mhzytz.py
# Source Nodes: [getattr_l__mod___layer3___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer3___0___pool => avg_pool2d_2
triton_poi_fused_avg_pool2d_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 14) % 14
    x0 = xindex % 14
    x3 = (xindex // 25088)
    x6 = (xindex // 14) % 1792
    x7 = xindex % 25088
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (301027 + (2*x0) + (56*x6) + (401408*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (301028 + (2*x0) + (56*x6) + (401408*x3)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (301029 + (2*x0) + (56*x6) + (401408*x3)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (301055 + (2*x0) + (56*x6) + (401408*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (301056 + (2*x0) + (56*x6) + (401408*x3)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (301057 + (2*x0) + (56*x6) + (401408*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (301083 + (2*x0) + (56*x6) + (401408*x3)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (301084 + (2*x0) + (56*x6) + (401408*x3)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (301085 + (2*x0) + (56*x6) + (401408*x3)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 29, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x7 + (100352*x3)), tmp145, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xn/cxnfnhvmnvm5u6qyshseyekqzixkr55o5d6ct2x4h55hfo5i4dyq.py
# Source Nodes: [sp_93, sp_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# sp_93 => add_96, mul_118, mul_119, sub_39
# sp_94 => relu_37
triton_poi_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 128
    x2 = (xindex // 25088)
    x4 = xindex % 25088
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x4 + (100352*x2)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qw/cqwqhfy2fkacnegpmsp2i5a4wysm3fi3p5cjipbhfdioydxjuajm.py
# Source Nodes: [out_61, out_62, shortcut_10, shortcut_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_61 => add_102, mul_127, mul_128, sub_42
# out_62 => add_105
# shortcut_10 => add_104, mul_130, mul_131, sub_43
# shortcut_11 => relu_40
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(in_out_ptr0 + (x3), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lf/clfkkgnq2agelsc2dszhyw2f2muy6v3tgrvh6gc2v4xde5tgtxwz.py
# Source Nodes: [out_65, out_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_65 => add_107, mul_133, mul_134, sub_44
# out_66 => relu_41
triton_poi_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yo/cyom7nhyegtxy4rhto6p5k6jv2pggotekrgg6sa3lvpzalz7wfoa.py
# Source Nodes: [sp_106, sp_107, sp_108, sp_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
# sp_106 => add_109, mul_136, mul_137, sub_45
# sp_107 => relu_42
# sp_108 => add_110
# sp_109 => convolution_46
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 128
    x2 = (xindex // 25088)
    x4 = xindex % 25088
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (25088 + x4 + (100352*x2)), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x4 + (100352*x2)), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xh/cxhtwq4opaobfdwrsqgmyvsrsskgri5awmsstsnc7t65zlqugfqo.py
# Source Nodes: [sp_110, sp_111, sp_112, sp_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
# sp_110 => add_112, mul_139, mul_140, sub_46
# sp_111 => relu_43
# sp_112 => add_113
# sp_113 => convolution_47
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 128
    x2 = (xindex // 25088)
    x4 = xindex % 25088
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (50176 + x4 + (100352*x2)), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x4 + (100352*x2)), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/go/cgomoihuvnb3tb5qbx7pzu7aov4amqlwlkzeytlyen7p2ajrexmp.py
# Source Nodes: [cat_23], Original ATen: [aten.cat]
# cat_23 => cat_8
triton_poi_fused_cat_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 25088
    x1 = (xindex // 25088)
    tmp0 = tl.load(in_ptr0 + (75264 + x0 + (100352*x1)), None)
    tl.store(out_ptr0 + (x0 + (100352*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gt/cgt4vo3k3mzz4a4glc54genhar4u2yhxnku2m5pmppcfe5vsaqmv.py
# Source Nodes: [out_69, out_70, shortcut_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_69 => add_117, mul_145, mul_146, sub_48
# out_70 => add_118
# shortcut_12 => relu_45
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pa/cpapxczerjhdjyd7dcujutpoak7yvgxmvawiyhttovj33xm643iy.py
# Source Nodes: [out_105, out_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_105 => add_172, mul_208, mul_209, sub_69
# out_106 => relu_66
triton_poi_fused__native_batch_norm_legit_no_training_relu_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3d/c3dutqx5o2pf33pmdwaxootstsfxokuc4tm3oxwjyohp7m4k3zm2.py
# Source Nodes: [getattr_l__mod___layer4___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer4___0___pool => avg_pool2d_3
triton_poi_fused_avg_pool2d_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 7) % 7
    x0 = xindex % 7
    x3 = (xindex // 12544)
    x6 = (xindex // 7) % 1792
    x7 = xindex % 12544
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (150513 + (2*x0) + (28*x6) + (200704*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (150514 + (2*x0) + (28*x6) + (200704*x3)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (150515 + (2*x0) + (28*x6) + (200704*x3)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (150527 + (2*x0) + (28*x6) + (200704*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (150528 + (2*x0) + (28*x6) + (200704*x3)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (150529 + (2*x0) + (28*x6) + (200704*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (150541 + (2*x0) + (28*x6) + (200704*x3)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (150542 + (2*x0) + (28*x6) + (200704*x3)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (150543 + (2*x0) + (28*x6) + (200704*x3)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 15, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x7 + (50176*x3)), tmp145, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ma/cmaxpoekv5mxwpogkxhxupszrq5i25j5i5bhpiyjbydhbdrtbkjj.py
# Source Nodes: [sp_171, sp_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# sp_171 => add_174, mul_211, mul_212, sub_70
# sp_172 => relu_67
triton_poi_fused__native_batch_norm_legit_no_training_relu_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 256
    x2 = (xindex // 12544)
    x4 = xindex % 12544
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x4 + (50176*x2)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ir/cirh5yqoimeez6wgwnscxtu5e53afad4delg2bvrcgbklghmz4qo.py
# Source Nodes: [out_109, out_110, shortcut_17, shortcut_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_109 => add_180, mul_220, mul_221, sub_73
# out_110 => add_183
# shortcut_17 => add_182, mul_223, mul_224, sub_74
# shortcut_18 => relu_70
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 2048
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(in_out_ptr0 + (x3), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lh/clhv5aqh6vjbfl5przggm4advxbowmj53l6mwhd6i6kazlevvmfj.py
# Source Nodes: [out_113, out_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_113 => add_185, mul_226, mul_227, sub_75
# out_114 => relu_71
triton_poi_fused__native_batch_norm_legit_no_training_relu_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1024
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yh/cyhghscbrxbfzvybcllghyrdvwdlm2jxrpkisxso6pztw44sum4r.py
# Source Nodes: [sp_184, sp_185, sp_186, sp_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
# sp_184 => add_187, mul_229, mul_230, sub_76
# sp_185 => relu_72
# sp_186 => add_188
# sp_187 => convolution_77
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 256
    x2 = (xindex // 12544)
    x4 = xindex % 12544
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (12544 + x4 + (50176*x2)), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x4 + (50176*x2)), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/26/c26tfhawdjrmid2vhkfwnohz55gcaplz53oddy7v6aujvej6fjiu.py
# Source Nodes: [sp_188, sp_189, sp_190, sp_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
# sp_188 => add_190, mul_232, mul_233, sub_77
# sp_189 => relu_73
# sp_190 => add_191
# sp_191 => convolution_78
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 256
    x2 = (xindex // 12544)
    x4 = xindex % 12544
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (25088 + x4 + (50176*x2)), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x4 + (50176*x2)), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/og/coghirpjucnfo3rx6sylhcwwwu7msg67ghrhxfvlwfdz2ts3x426.py
# Source Nodes: [cat_17], Original ATen: [aten.cat]
# cat_17 => cat_14
triton_poi_fused_cat_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 12544
    x1 = (xindex // 12544)
    tmp0 = tl.load(in_ptr0 + (37632 + x0 + (50176*x1)), None)
    tl.store(out_ptr0 + (x0 + (50176*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2h/c2hnqwvnln2o4klfpnkfezo44el6reyhxk4x74ssboe7ix5e75ud.py
# Source Nodes: [out_117, out_118, shortcut_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_117 => add_195, mul_238, mul_239, sub_79
# out_118 => add_196
# shortcut_19 => relu_75
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vp/cvpf3hukzykeow4mtyquatdvz5wupus454eyimuv4x3xm6ik72pq.py
# Source Nodes: [out_125, out_126, x_8, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
# out_125 => add_208, mul_253, mul_254, sub_84
# out_126 => add_209
# x_8 => relu_80
# x_9 => mean
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_37', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (r2 + (49*x3)), rmask, other=0.0)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = 49.0
    tmp23 = tmp21 / tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg7_1, (32, ), (1, ))
    assert_size_stride(arg8_1, (32, ), (1, ))
    assert_size_stride(arg9_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg10_1, (32, ), (1, ))
    assert_size_stride(arg11_1, (32, ), (1, ))
    assert_size_stride(arg12_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg13_1, (32, ), (1, ))
    assert_size_stride(arg14_1, (32, ), (1, ))
    assert_size_stride(arg15_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (256, ), (1, ))
    assert_size_stride(arg21_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg22_1, (128, ), (1, ))
    assert_size_stride(arg23_1, (128, ), (1, ))
    assert_size_stride(arg24_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg25_1, (32, ), (1, ))
    assert_size_stride(arg26_1, (32, ), (1, ))
    assert_size_stride(arg27_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg28_1, (32, ), (1, ))
    assert_size_stride(arg29_1, (32, ), (1, ))
    assert_size_stride(arg30_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg31_1, (32, ), (1, ))
    assert_size_stride(arg32_1, (32, ), (1, ))
    assert_size_stride(arg33_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg40_1, (32, ), (1, ))
    assert_size_stride(arg41_1, (32, ), (1, ))
    assert_size_stride(arg42_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg43_1, (32, ), (1, ))
    assert_size_stride(arg44_1, (32, ), (1, ))
    assert_size_stride(arg45_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg46_1, (32, ), (1, ))
    assert_size_stride(arg47_1, (32, ), (1, ))
    assert_size_stride(arg48_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg55_1, (64, ), (1, ))
    assert_size_stride(arg56_1, (64, ), (1, ))
    assert_size_stride(arg57_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg58_1, (64, ), (1, ))
    assert_size_stride(arg59_1, (64, ), (1, ))
    assert_size_stride(arg60_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg61_1, (64, ), (1, ))
    assert_size_stride(arg62_1, (64, ), (1, ))
    assert_size_stride(arg63_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg64_1, (512, ), (1, ))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg67_1, (512, ), (1, ))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg70_1, (256, ), (1, ))
    assert_size_stride(arg71_1, (256, ), (1, ))
    assert_size_stride(arg72_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg73_1, (64, ), (1, ))
    assert_size_stride(arg74_1, (64, ), (1, ))
    assert_size_stride(arg75_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg76_1, (64, ), (1, ))
    assert_size_stride(arg77_1, (64, ), (1, ))
    assert_size_stride(arg78_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg79_1, (64, ), (1, ))
    assert_size_stride(arg80_1, (64, ), (1, ))
    assert_size_stride(arg81_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg82_1, (512, ), (1, ))
    assert_size_stride(arg83_1, (512, ), (1, ))
    assert_size_stride(arg84_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg85_1, (256, ), (1, ))
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg88_1, (64, ), (1, ))
    assert_size_stride(arg89_1, (64, ), (1, ))
    assert_size_stride(arg90_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg91_1, (64, ), (1, ))
    assert_size_stride(arg92_1, (64, ), (1, ))
    assert_size_stride(arg93_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg94_1, (64, ), (1, ))
    assert_size_stride(arg95_1, (64, ), (1, ))
    assert_size_stride(arg96_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg97_1, (512, ), (1, ))
    assert_size_stride(arg98_1, (512, ), (1, ))
    assert_size_stride(arg99_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg100_1, (256, ), (1, ))
    assert_size_stride(arg101_1, (256, ), (1, ))
    assert_size_stride(arg102_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg103_1, (64, ), (1, ))
    assert_size_stride(arg104_1, (64, ), (1, ))
    assert_size_stride(arg105_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg106_1, (64, ), (1, ))
    assert_size_stride(arg107_1, (64, ), (1, ))
    assert_size_stride(arg108_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg109_1, (64, ), (1, ))
    assert_size_stride(arg110_1, (64, ), (1, ))
    assert_size_stride(arg111_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg112_1, (512, ), (1, ))
    assert_size_stride(arg113_1, (512, ), (1, ))
    assert_size_stride(arg114_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (512, ), (1, ))
    assert_size_stride(arg117_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg118_1, (128, ), (1, ))
    assert_size_stride(arg119_1, (128, ), (1, ))
    assert_size_stride(arg120_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg121_1, (128, ), (1, ))
    assert_size_stride(arg122_1, (128, ), (1, ))
    assert_size_stride(arg123_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg124_1, (128, ), (1, ))
    assert_size_stride(arg125_1, (128, ), (1, ))
    assert_size_stride(arg126_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg127_1, (1024, ), (1, ))
    assert_size_stride(arg128_1, (1024, ), (1, ))
    assert_size_stride(arg129_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg130_1, (1024, ), (1, ))
    assert_size_stride(arg131_1, (1024, ), (1, ))
    assert_size_stride(arg132_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg133_1, (512, ), (1, ))
    assert_size_stride(arg134_1, (512, ), (1, ))
    assert_size_stride(arg135_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg136_1, (128, ), (1, ))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg139_1, (128, ), (1, ))
    assert_size_stride(arg140_1, (128, ), (1, ))
    assert_size_stride(arg141_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg142_1, (128, ), (1, ))
    assert_size_stride(arg143_1, (128, ), (1, ))
    assert_size_stride(arg144_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg145_1, (1024, ), (1, ))
    assert_size_stride(arg146_1, (1024, ), (1, ))
    assert_size_stride(arg147_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg148_1, (512, ), (1, ))
    assert_size_stride(arg149_1, (512, ), (1, ))
    assert_size_stride(arg150_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg151_1, (128, ), (1, ))
    assert_size_stride(arg152_1, (128, ), (1, ))
    assert_size_stride(arg153_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg154_1, (128, ), (1, ))
    assert_size_stride(arg155_1, (128, ), (1, ))
    assert_size_stride(arg156_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg157_1, (128, ), (1, ))
    assert_size_stride(arg158_1, (128, ), (1, ))
    assert_size_stride(arg159_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg160_1, (1024, ), (1, ))
    assert_size_stride(arg161_1, (1024, ), (1, ))
    assert_size_stride(arg162_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg163_1, (512, ), (1, ))
    assert_size_stride(arg164_1, (512, ), (1, ))
    assert_size_stride(arg165_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg166_1, (128, ), (1, ))
    assert_size_stride(arg167_1, (128, ), (1, ))
    assert_size_stride(arg168_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg169_1, (128, ), (1, ))
    assert_size_stride(arg170_1, (128, ), (1, ))
    assert_size_stride(arg171_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg172_1, (128, ), (1, ))
    assert_size_stride(arg173_1, (128, ), (1, ))
    assert_size_stride(arg174_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg175_1, (1024, ), (1, ))
    assert_size_stride(arg176_1, (1024, ), (1, ))
    assert_size_stride(arg177_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg178_1, (512, ), (1, ))
    assert_size_stride(arg179_1, (512, ), (1, ))
    assert_size_stride(arg180_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg181_1, (128, ), (1, ))
    assert_size_stride(arg182_1, (128, ), (1, ))
    assert_size_stride(arg183_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg184_1, (128, ), (1, ))
    assert_size_stride(arg185_1, (128, ), (1, ))
    assert_size_stride(arg186_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg187_1, (128, ), (1, ))
    assert_size_stride(arg188_1, (128, ), (1, ))
    assert_size_stride(arg189_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg190_1, (1024, ), (1, ))
    assert_size_stride(arg191_1, (1024, ), (1, ))
    assert_size_stride(arg192_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg193_1, (512, ), (1, ))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg196_1, (128, ), (1, ))
    assert_size_stride(arg197_1, (128, ), (1, ))
    assert_size_stride(arg198_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg199_1, (128, ), (1, ))
    assert_size_stride(arg200_1, (128, ), (1, ))
    assert_size_stride(arg201_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg202_1, (128, ), (1, ))
    assert_size_stride(arg203_1, (128, ), (1, ))
    assert_size_stride(arg204_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg205_1, (1024, ), (1, ))
    assert_size_stride(arg206_1, (1024, ), (1, ))
    assert_size_stride(arg207_1, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg208_1, (1024, ), (1, ))
    assert_size_stride(arg209_1, (1024, ), (1, ))
    assert_size_stride(arg210_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg211_1, (256, ), (1, ))
    assert_size_stride(arg212_1, (256, ), (1, ))
    assert_size_stride(arg213_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg214_1, (256, ), (1, ))
    assert_size_stride(arg215_1, (256, ), (1, ))
    assert_size_stride(arg216_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg217_1, (256, ), (1, ))
    assert_size_stride(arg218_1, (256, ), (1, ))
    assert_size_stride(arg219_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg220_1, (2048, ), (1, ))
    assert_size_stride(arg221_1, (2048, ), (1, ))
    assert_size_stride(arg222_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg223_1, (2048, ), (1, ))
    assert_size_stride(arg224_1, (2048, ), (1, ))
    assert_size_stride(arg225_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg226_1, (1024, ), (1, ))
    assert_size_stride(arg227_1, (1024, ), (1, ))
    assert_size_stride(arg228_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg229_1, (256, ), (1, ))
    assert_size_stride(arg230_1, (256, ), (1, ))
    assert_size_stride(arg231_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg232_1, (256, ), (1, ))
    assert_size_stride(arg233_1, (256, ), (1, ))
    assert_size_stride(arg234_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg235_1, (256, ), (1, ))
    assert_size_stride(arg236_1, (256, ), (1, ))
    assert_size_stride(arg237_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg238_1, (2048, ), (1, ))
    assert_size_stride(arg239_1, (2048, ), (1, ))
    assert_size_stride(arg240_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg241_1, (1024, ), (1, ))
    assert_size_stride(arg242_1, (1024, ), (1, ))
    assert_size_stride(arg243_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg244_1, (256, ), (1, ))
    assert_size_stride(arg245_1, (256, ), (1, ))
    assert_size_stride(arg246_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg247_1, (256, ), (1, ))
    assert_size_stride(arg248_1, (256, ), (1, ))
    assert_size_stride(arg249_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg250_1, (256, ), (1, ))
    assert_size_stride(arg251_1, (256, ), (1, ))
    assert_size_stride(arg252_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg253_1, (2048, ), (1, ))
    assert_size_stride(arg254_1, (2048, ), (1, ))
    assert_size_stride(arg255_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg256_1, (1000, ), (1, ))
    assert_size_stride(arg257_1, (64, ), (1, ))
    assert_size_stride(arg258_1, (64, ), (1, ))
    assert_size_stride(arg259_1, (), ())
    assert_size_stride(arg260_1, (128, ), (1, ))
    assert_size_stride(arg261_1, (128, ), (1, ))
    assert_size_stride(arg262_1, (), ())
    assert_size_stride(arg263_1, (32, ), (1, ))
    assert_size_stride(arg264_1, (32, ), (1, ))
    assert_size_stride(arg265_1, (), ())
    assert_size_stride(arg266_1, (32, ), (1, ))
    assert_size_stride(arg267_1, (32, ), (1, ))
    assert_size_stride(arg268_1, (), ())
    assert_size_stride(arg269_1, (32, ), (1, ))
    assert_size_stride(arg270_1, (32, ), (1, ))
    assert_size_stride(arg271_1, (), ())
    assert_size_stride(arg272_1, (256, ), (1, ))
    assert_size_stride(arg273_1, (256, ), (1, ))
    assert_size_stride(arg274_1, (), ())
    assert_size_stride(arg275_1, (256, ), (1, ))
    assert_size_stride(arg276_1, (256, ), (1, ))
    assert_size_stride(arg277_1, (), ())
    assert_size_stride(arg278_1, (128, ), (1, ))
    assert_size_stride(arg279_1, (128, ), (1, ))
    assert_size_stride(arg280_1, (), ())
    assert_size_stride(arg281_1, (32, ), (1, ))
    assert_size_stride(arg282_1, (32, ), (1, ))
    assert_size_stride(arg283_1, (), ())
    assert_size_stride(arg284_1, (32, ), (1, ))
    assert_size_stride(arg285_1, (32, ), (1, ))
    assert_size_stride(arg286_1, (), ())
    assert_size_stride(arg287_1, (32, ), (1, ))
    assert_size_stride(arg288_1, (32, ), (1, ))
    assert_size_stride(arg289_1, (), ())
    assert_size_stride(arg290_1, (256, ), (1, ))
    assert_size_stride(arg291_1, (256, ), (1, ))
    assert_size_stride(arg292_1, (), ())
    assert_size_stride(arg293_1, (128, ), (1, ))
    assert_size_stride(arg294_1, (128, ), (1, ))
    assert_size_stride(arg295_1, (), ())
    assert_size_stride(arg296_1, (32, ), (1, ))
    assert_size_stride(arg297_1, (32, ), (1, ))
    assert_size_stride(arg298_1, (), ())
    assert_size_stride(arg299_1, (32, ), (1, ))
    assert_size_stride(arg300_1, (32, ), (1, ))
    assert_size_stride(arg301_1, (), ())
    assert_size_stride(arg302_1, (32, ), (1, ))
    assert_size_stride(arg303_1, (32, ), (1, ))
    assert_size_stride(arg304_1, (), ())
    assert_size_stride(arg305_1, (256, ), (1, ))
    assert_size_stride(arg306_1, (256, ), (1, ))
    assert_size_stride(arg307_1, (), ())
    assert_size_stride(arg308_1, (256, ), (1, ))
    assert_size_stride(arg309_1, (256, ), (1, ))
    assert_size_stride(arg310_1, (), ())
    assert_size_stride(arg311_1, (64, ), (1, ))
    assert_size_stride(arg312_1, (64, ), (1, ))
    assert_size_stride(arg313_1, (), ())
    assert_size_stride(arg314_1, (64, ), (1, ))
    assert_size_stride(arg315_1, (64, ), (1, ))
    assert_size_stride(arg316_1, (), ())
    assert_size_stride(arg317_1, (64, ), (1, ))
    assert_size_stride(arg318_1, (64, ), (1, ))
    assert_size_stride(arg319_1, (), ())
    assert_size_stride(arg320_1, (512, ), (1, ))
    assert_size_stride(arg321_1, (512, ), (1, ))
    assert_size_stride(arg322_1, (), ())
    assert_size_stride(arg323_1, (512, ), (1, ))
    assert_size_stride(arg324_1, (512, ), (1, ))
    assert_size_stride(arg325_1, (), ())
    assert_size_stride(arg326_1, (256, ), (1, ))
    assert_size_stride(arg327_1, (256, ), (1, ))
    assert_size_stride(arg328_1, (), ())
    assert_size_stride(arg329_1, (64, ), (1, ))
    assert_size_stride(arg330_1, (64, ), (1, ))
    assert_size_stride(arg331_1, (), ())
    assert_size_stride(arg332_1, (64, ), (1, ))
    assert_size_stride(arg333_1, (64, ), (1, ))
    assert_size_stride(arg334_1, (), ())
    assert_size_stride(arg335_1, (64, ), (1, ))
    assert_size_stride(arg336_1, (64, ), (1, ))
    assert_size_stride(arg337_1, (), ())
    assert_size_stride(arg338_1, (512, ), (1, ))
    assert_size_stride(arg339_1, (512, ), (1, ))
    assert_size_stride(arg340_1, (), ())
    assert_size_stride(arg341_1, (256, ), (1, ))
    assert_size_stride(arg342_1, (256, ), (1, ))
    assert_size_stride(arg343_1, (), ())
    assert_size_stride(arg344_1, (64, ), (1, ))
    assert_size_stride(arg345_1, (64, ), (1, ))
    assert_size_stride(arg346_1, (), ())
    assert_size_stride(arg347_1, (64, ), (1, ))
    assert_size_stride(arg348_1, (64, ), (1, ))
    assert_size_stride(arg349_1, (), ())
    assert_size_stride(arg350_1, (64, ), (1, ))
    assert_size_stride(arg351_1, (64, ), (1, ))
    assert_size_stride(arg352_1, (), ())
    assert_size_stride(arg353_1, (512, ), (1, ))
    assert_size_stride(arg354_1, (512, ), (1, ))
    assert_size_stride(arg355_1, (), ())
    assert_size_stride(arg356_1, (256, ), (1, ))
    assert_size_stride(arg357_1, (256, ), (1, ))
    assert_size_stride(arg358_1, (), ())
    assert_size_stride(arg359_1, (64, ), (1, ))
    assert_size_stride(arg360_1, (64, ), (1, ))
    assert_size_stride(arg361_1, (), ())
    assert_size_stride(arg362_1, (64, ), (1, ))
    assert_size_stride(arg363_1, (64, ), (1, ))
    assert_size_stride(arg364_1, (), ())
    assert_size_stride(arg365_1, (64, ), (1, ))
    assert_size_stride(arg366_1, (64, ), (1, ))
    assert_size_stride(arg367_1, (), ())
    assert_size_stride(arg368_1, (512, ), (1, ))
    assert_size_stride(arg369_1, (512, ), (1, ))
    assert_size_stride(arg370_1, (), ())
    assert_size_stride(arg371_1, (512, ), (1, ))
    assert_size_stride(arg372_1, (512, ), (1, ))
    assert_size_stride(arg373_1, (), ())
    assert_size_stride(arg374_1, (128, ), (1, ))
    assert_size_stride(arg375_1, (128, ), (1, ))
    assert_size_stride(arg376_1, (), ())
    assert_size_stride(arg377_1, (128, ), (1, ))
    assert_size_stride(arg378_1, (128, ), (1, ))
    assert_size_stride(arg379_1, (), ())
    assert_size_stride(arg380_1, (128, ), (1, ))
    assert_size_stride(arg381_1, (128, ), (1, ))
    assert_size_stride(arg382_1, (), ())
    assert_size_stride(arg383_1, (1024, ), (1, ))
    assert_size_stride(arg384_1, (1024, ), (1, ))
    assert_size_stride(arg385_1, (), ())
    assert_size_stride(arg386_1, (1024, ), (1, ))
    assert_size_stride(arg387_1, (1024, ), (1, ))
    assert_size_stride(arg388_1, (), ())
    assert_size_stride(arg389_1, (512, ), (1, ))
    assert_size_stride(arg390_1, (512, ), (1, ))
    assert_size_stride(arg391_1, (), ())
    assert_size_stride(arg392_1, (128, ), (1, ))
    assert_size_stride(arg393_1, (128, ), (1, ))
    assert_size_stride(arg394_1, (), ())
    assert_size_stride(arg395_1, (128, ), (1, ))
    assert_size_stride(arg396_1, (128, ), (1, ))
    assert_size_stride(arg397_1, (), ())
    assert_size_stride(arg398_1, (128, ), (1, ))
    assert_size_stride(arg399_1, (128, ), (1, ))
    assert_size_stride(arg400_1, (), ())
    assert_size_stride(arg401_1, (1024, ), (1, ))
    assert_size_stride(arg402_1, (1024, ), (1, ))
    assert_size_stride(arg403_1, (), ())
    assert_size_stride(arg404_1, (512, ), (1, ))
    assert_size_stride(arg405_1, (512, ), (1, ))
    assert_size_stride(arg406_1, (), ())
    assert_size_stride(arg407_1, (128, ), (1, ))
    assert_size_stride(arg408_1, (128, ), (1, ))
    assert_size_stride(arg409_1, (), ())
    assert_size_stride(arg410_1, (128, ), (1, ))
    assert_size_stride(arg411_1, (128, ), (1, ))
    assert_size_stride(arg412_1, (), ())
    assert_size_stride(arg413_1, (128, ), (1, ))
    assert_size_stride(arg414_1, (128, ), (1, ))
    assert_size_stride(arg415_1, (), ())
    assert_size_stride(arg416_1, (1024, ), (1, ))
    assert_size_stride(arg417_1, (1024, ), (1, ))
    assert_size_stride(arg418_1, (), ())
    assert_size_stride(arg419_1, (512, ), (1, ))
    assert_size_stride(arg420_1, (512, ), (1, ))
    assert_size_stride(arg421_1, (), ())
    assert_size_stride(arg422_1, (128, ), (1, ))
    assert_size_stride(arg423_1, (128, ), (1, ))
    assert_size_stride(arg424_1, (), ())
    assert_size_stride(arg425_1, (128, ), (1, ))
    assert_size_stride(arg426_1, (128, ), (1, ))
    assert_size_stride(arg427_1, (), ())
    assert_size_stride(arg428_1, (128, ), (1, ))
    assert_size_stride(arg429_1, (128, ), (1, ))
    assert_size_stride(arg430_1, (), ())
    assert_size_stride(arg431_1, (1024, ), (1, ))
    assert_size_stride(arg432_1, (1024, ), (1, ))
    assert_size_stride(arg433_1, (), ())
    assert_size_stride(arg434_1, (512, ), (1, ))
    assert_size_stride(arg435_1, (512, ), (1, ))
    assert_size_stride(arg436_1, (), ())
    assert_size_stride(arg437_1, (128, ), (1, ))
    assert_size_stride(arg438_1, (128, ), (1, ))
    assert_size_stride(arg439_1, (), ())
    assert_size_stride(arg440_1, (128, ), (1, ))
    assert_size_stride(arg441_1, (128, ), (1, ))
    assert_size_stride(arg442_1, (), ())
    assert_size_stride(arg443_1, (128, ), (1, ))
    assert_size_stride(arg444_1, (128, ), (1, ))
    assert_size_stride(arg445_1, (), ())
    assert_size_stride(arg446_1, (1024, ), (1, ))
    assert_size_stride(arg447_1, (1024, ), (1, ))
    assert_size_stride(arg448_1, (), ())
    assert_size_stride(arg449_1, (512, ), (1, ))
    assert_size_stride(arg450_1, (512, ), (1, ))
    assert_size_stride(arg451_1, (), ())
    assert_size_stride(arg452_1, (128, ), (1, ))
    assert_size_stride(arg453_1, (128, ), (1, ))
    assert_size_stride(arg454_1, (), ())
    assert_size_stride(arg455_1, (128, ), (1, ))
    assert_size_stride(arg456_1, (128, ), (1, ))
    assert_size_stride(arg457_1, (), ())
    assert_size_stride(arg458_1, (128, ), (1, ))
    assert_size_stride(arg459_1, (128, ), (1, ))
    assert_size_stride(arg460_1, (), ())
    assert_size_stride(arg461_1, (1024, ), (1, ))
    assert_size_stride(arg462_1, (1024, ), (1, ))
    assert_size_stride(arg463_1, (), ())
    assert_size_stride(arg464_1, (1024, ), (1, ))
    assert_size_stride(arg465_1, (1024, ), (1, ))
    assert_size_stride(arg466_1, (), ())
    assert_size_stride(arg467_1, (256, ), (1, ))
    assert_size_stride(arg468_1, (256, ), (1, ))
    assert_size_stride(arg469_1, (), ())
    assert_size_stride(arg470_1, (256, ), (1, ))
    assert_size_stride(arg471_1, (256, ), (1, ))
    assert_size_stride(arg472_1, (), ())
    assert_size_stride(arg473_1, (256, ), (1, ))
    assert_size_stride(arg474_1, (256, ), (1, ))
    assert_size_stride(arg475_1, (), ())
    assert_size_stride(arg476_1, (2048, ), (1, ))
    assert_size_stride(arg477_1, (2048, ), (1, ))
    assert_size_stride(arg478_1, (), ())
    assert_size_stride(arg479_1, (2048, ), (1, ))
    assert_size_stride(arg480_1, (2048, ), (1, ))
    assert_size_stride(arg481_1, (), ())
    assert_size_stride(arg482_1, (1024, ), (1, ))
    assert_size_stride(arg483_1, (1024, ), (1, ))
    assert_size_stride(arg484_1, (), ())
    assert_size_stride(arg485_1, (256, ), (1, ))
    assert_size_stride(arg486_1, (256, ), (1, ))
    assert_size_stride(arg487_1, (), ())
    assert_size_stride(arg488_1, (256, ), (1, ))
    assert_size_stride(arg489_1, (256, ), (1, ))
    assert_size_stride(arg490_1, (), ())
    assert_size_stride(arg491_1, (256, ), (1, ))
    assert_size_stride(arg492_1, (256, ), (1, ))
    assert_size_stride(arg493_1, (), ())
    assert_size_stride(arg494_1, (2048, ), (1, ))
    assert_size_stride(arg495_1, (2048, ), (1, ))
    assert_size_stride(arg496_1, (), ())
    assert_size_stride(arg497_1, (1024, ), (1, ))
    assert_size_stride(arg498_1, (1024, ), (1, ))
    assert_size_stride(arg499_1, (), ())
    assert_size_stride(arg500_1, (256, ), (1, ))
    assert_size_stride(arg501_1, (256, ), (1, ))
    assert_size_stride(arg502_1, (), ())
    assert_size_stride(arg503_1, (256, ), (1, ))
    assert_size_stride(arg504_1, (256, ), (1, ))
    assert_size_stride(arg505_1, (), ())
    assert_size_stride(arg506_1, (256, ), (1, ))
    assert_size_stride(arg507_1, (256, ), (1, ))
    assert_size_stride(arg508_1, (), ())
    assert_size_stride(arg509_1, (2048, ), (1, ))
    assert_size_stride(arg510_1, (2048, ), (1, ))
    assert_size_stride(arg511_1, (), ())
    assert_size_stride(arg512_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg512_1, arg0_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 64, 112, 112), (802816, 12544, 112, 1))
        del arg0_1
        del arg512_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf1, arg257_1, arg258_1, arg1_1, arg2_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg1_1
        del arg257_1
        del arg258_1
        del arg2_1
        buf2 = empty((8, 64, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1.run(buf1, buf2, 1605632, grid=grid(1605632), stream=stream0)
        del buf1
        # Source Nodes: [out], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, arg3_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg3_1
        buf4 = buf3; del buf3  # reuse
        # Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf4, arg260_1, arg261_1, arg4_1, arg5_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg260_1
        del arg261_1
        del arg4_1
        del arg5_1
        # Source Nodes: [sp_1], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(reinterpret_tensor(buf4, (8, 32, 56, 56), (401408, 3136, 56, 1), 0), arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf5, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg6_1
        # Source Nodes: [sp_5], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(reinterpret_tensor(buf4, (8, 32, 56, 56), (401408, 3136, 56, 1), 100352), arg9_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf6, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg9_1
        # Source Nodes: [sp_9], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(reinterpret_tensor(buf4, (8, 32, 56, 56), (401408, 3136, 56, 1), 200704), arg12_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf7, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg12_1
        buf12 = empty((8, 128, 56, 56), device='cuda', dtype=torch.float32)
        buf8 = reinterpret_tensor(buf12, (8, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
        # Source Nodes: [getattr_l__mod___layer1___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_3.run(buf4, buf8, 802816, grid=grid(802816), stream=stream0)
        del buf4
        buf9 = reinterpret_tensor(buf12, (8, 32, 56, 56), (401408, 3136, 56, 1), 0)  # alias
        # Source Nodes: [sp_2, sp_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf5, arg263_1, arg264_1, arg7_1, arg8_1, buf9, 802816, grid=grid(802816), stream=stream0)
        del arg263_1
        del arg264_1
        del arg7_1
        del arg8_1
        del buf5
        buf10 = reinterpret_tensor(buf12, (8, 32, 56, 56), (401408, 3136, 56, 1), 100352)  # alias
        # Source Nodes: [sp_6, sp_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf6, arg266_1, arg267_1, arg10_1, arg11_1, buf10, 802816, grid=grid(802816), stream=stream0)
        del arg10_1
        del arg11_1
        del arg266_1
        del arg267_1
        del buf6
        buf11 = reinterpret_tensor(buf12, (8, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
        # Source Nodes: [sp_10, sp_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf7, arg269_1, arg270_1, arg13_1, arg14_1, buf11, 802816, grid=grid(802816), stream=stream0)
        del arg13_1
        del arg14_1
        del arg269_1
        del arg270_1
        del buf10
        del buf11
        del buf8
        del buf9
        # Source Nodes: [out_4], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg15_1
        # Source Nodes: [getattr_l__mod___layer1___0___downsample_0], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf2, arg18_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg18_1
        buf15 = buf13; del buf13  # reuse
        buf16 = buf15; del buf15  # reuse
        # Source Nodes: [out_5, out_6, shortcut_1, shortcut_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_5.run(buf16, arg272_1, arg273_1, arg16_1, arg17_1, buf14, arg275_1, arg276_1, arg19_1, arg20_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg16_1
        del arg17_1
        del arg19_1
        del arg20_1
        del arg272_1
        del arg273_1
        del arg275_1
        del arg276_1
        del buf14
        # Source Nodes: [out_8], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg21_1
        buf18 = buf17; del buf17  # reuse
        # Source Nodes: [out_10, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf18, arg278_1, arg279_1, arg22_1, arg23_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg22_1
        del arg23_1
        del arg278_1
        del arg279_1
        # Source Nodes: [sp_14], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(reinterpret_tensor(buf18, (8, 32, 56, 56), (401408, 3136, 56, 1), 0), arg24_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf19, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg24_1
        buf28 = buf12; del buf12  # reuse
        buf20 = reinterpret_tensor(buf28, (8, 32, 56, 56), (401408, 3136, 56, 1), 0)  # alias
        buf21 = buf7; del buf7  # reuse
        # Source Nodes: [sp_15, sp_16, sp_17, sp_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_6.run(buf19, arg281_1, arg282_1, arg25_1, arg26_1, buf18, buf20, buf21, 802816, grid=grid(802816), stream=stream0)
        del arg25_1
        del arg26_1
        del arg281_1
        del arg282_1
        del buf19
        # Source Nodes: [sp_17, sp_18], Original ATen: [aten.add, aten.convolution]
        buf22 = extern_kernels.convolution(buf21, arg27_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf22, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg27_1
        buf23 = reinterpret_tensor(buf28, (8, 32, 56, 56), (401408, 3136, 56, 1), 100352)  # alias
        buf24 = buf21; del buf21  # reuse
        # Source Nodes: [sp_19, sp_20, sp_21, sp_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_7.run(buf22, arg284_1, arg285_1, arg28_1, arg29_1, buf18, buf23, buf24, 802816, grid=grid(802816), stream=stream0)
        del arg284_1
        del arg285_1
        del arg28_1
        del arg29_1
        del buf22
        # Source Nodes: [sp_21, sp_22], Original ATen: [aten.add, aten.convolution]
        buf25 = extern_kernels.convolution(buf24, arg30_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf25, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg30_1
        del buf24
        buf26 = reinterpret_tensor(buf28, (8, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
        # Source Nodes: [sp_23, sp_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf25, arg287_1, arg288_1, arg31_1, arg32_1, buf26, 802816, grid=grid(802816), stream=stream0)
        del arg287_1
        del arg288_1
        del arg31_1
        del arg32_1
        buf27 = reinterpret_tensor(buf28, (8, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
        # Source Nodes: [cat_30], Original ATen: [aten.cat]
        triton_poi_fused_cat_8.run(buf18, buf27, 802816, grid=grid(802816), stream=stream0)
        del buf18
        del buf20
        del buf23
        del buf26
        del buf27
        # Source Nodes: [out_12], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg33_1
        buf30 = buf16; del buf16  # reuse
        # Source Nodes: [out_13, out_14, shortcut_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9.run(buf30, buf29, arg290_1, arg291_1, arg34_1, arg35_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg290_1
        del arg291_1
        del arg34_1
        del arg35_1
        del buf29
        # Source Nodes: [out_16], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg36_1
        buf32 = buf31; del buf31  # reuse
        # Source Nodes: [out_17, out_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf32, arg293_1, arg294_1, arg37_1, arg38_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg293_1
        del arg294_1
        del arg37_1
        del arg38_1
        # Source Nodes: [sp_27], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(reinterpret_tensor(buf32, (8, 32, 56, 56), (401408, 3136, 56, 1), 0), arg39_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf33, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg39_1
        buf42 = buf28; del buf28  # reuse
        buf34 = reinterpret_tensor(buf42, (8, 32, 56, 56), (401408, 3136, 56, 1), 0)  # alias
        buf35 = buf25; del buf25  # reuse
        # Source Nodes: [sp_28, sp_29, sp_30, sp_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_6.run(buf33, arg296_1, arg297_1, arg40_1, arg41_1, buf32, buf34, buf35, 802816, grid=grid(802816), stream=stream0)
        del arg296_1
        del arg297_1
        del arg40_1
        del arg41_1
        del buf33
        # Source Nodes: [sp_30, sp_31], Original ATen: [aten.add, aten.convolution]
        buf36 = extern_kernels.convolution(buf35, arg42_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf36, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg42_1
        buf37 = reinterpret_tensor(buf42, (8, 32, 56, 56), (401408, 3136, 56, 1), 100352)  # alias
        buf38 = buf35; del buf35  # reuse
        # Source Nodes: [sp_32, sp_33, sp_34, sp_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_7.run(buf36, arg299_1, arg300_1, arg43_1, arg44_1, buf32, buf37, buf38, 802816, grid=grid(802816), stream=stream0)
        del arg299_1
        del arg300_1
        del arg43_1
        del arg44_1
        del buf36
        # Source Nodes: [sp_34, sp_35], Original ATen: [aten.add, aten.convolution]
        buf39 = extern_kernels.convolution(buf38, arg45_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf39, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg45_1
        del buf38
        buf40 = reinterpret_tensor(buf42, (8, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
        # Source Nodes: [sp_36, sp_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf39, arg302_1, arg303_1, arg46_1, arg47_1, buf40, 802816, grid=grid(802816), stream=stream0)
        del arg302_1
        del arg303_1
        del arg46_1
        del arg47_1
        buf41 = reinterpret_tensor(buf42, (8, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
        # Source Nodes: [cat_29], Original ATen: [aten.cat]
        triton_poi_fused_cat_8.run(buf32, buf41, 802816, grid=grid(802816), stream=stream0)
        del buf32
        del buf34
        del buf37
        del buf40
        del buf41
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, arg48_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg48_1
        del buf42
        buf44 = buf30; del buf30  # reuse
        # Source Nodes: [out_21, out_22, shortcut_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9.run(buf44, buf43, arg305_1, arg306_1, arg49_1, arg50_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg305_1
        del arg306_1
        del arg49_1
        del arg50_1
        del buf43
        # Source Nodes: [out_24], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg51_1
        buf46 = buf45; del buf45  # reuse
        # Source Nodes: [out_25, out_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf46, arg308_1, arg309_1, arg52_1, arg53_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg308_1
        del arg309_1
        del arg52_1
        del arg53_1
        # Source Nodes: [sp_40], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(reinterpret_tensor(buf46, (8, 64, 56, 56), (802816, 3136, 56, 1), 0), arg54_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf47, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg54_1
        # Source Nodes: [sp_44], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(reinterpret_tensor(buf46, (8, 64, 56, 56), (802816, 3136, 56, 1), 200704), arg57_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf48, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg57_1
        # Source Nodes: [sp_48], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(reinterpret_tensor(buf46, (8, 64, 56, 56), (802816, 3136, 56, 1), 401408), arg60_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf49, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg60_1
        buf54 = reinterpret_tensor(buf2, (8, 256, 28, 28), (200704, 784, 28, 1), 0); del buf2  # reuse
        buf50 = reinterpret_tensor(buf54, (8, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        # Source Nodes: [getattr_l__mod___layer2___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_11.run(buf46, buf50, 401408, grid=grid(401408), stream=stream0)
        del buf46
        buf51 = reinterpret_tensor(buf54, (8, 64, 28, 28), (200704, 784, 28, 1), 0)  # alias
        # Source Nodes: [sp_41, sp_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf47, arg311_1, arg312_1, arg55_1, arg56_1, buf51, 401408, grid=grid(401408), stream=stream0)
        del arg311_1
        del arg312_1
        del arg55_1
        del arg56_1
        del buf47
        buf52 = reinterpret_tensor(buf54, (8, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        # Source Nodes: [sp_45, sp_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf48, arg314_1, arg315_1, arg58_1, arg59_1, buf52, 401408, grid=grid(401408), stream=stream0)
        del arg314_1
        del arg315_1
        del arg58_1
        del arg59_1
        del buf48
        buf53 = reinterpret_tensor(buf54, (8, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        # Source Nodes: [sp_49, sp_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf49, arg317_1, arg318_1, arg61_1, arg62_1, buf53, 401408, grid=grid(401408), stream=stream0)
        del arg317_1
        del arg318_1
        del arg61_1
        del arg62_1
        del buf50
        del buf51
        del buf52
        del buf53
        # Source Nodes: [out_28], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, arg63_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg63_1
        # Source Nodes: [getattr_l__mod___layer2___0___downsample_0], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf44, arg66_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg66_1
        del buf44
        buf57 = buf55; del buf55  # reuse
        buf58 = buf57; del buf57  # reuse
        # Source Nodes: [out_29, out_30, shortcut_5, shortcut_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13.run(buf58, arg320_1, arg321_1, arg64_1, arg65_1, buf56, arg323_1, arg324_1, arg67_1, arg68_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg320_1
        del arg321_1
        del arg323_1
        del arg324_1
        del arg64_1
        del arg65_1
        del arg67_1
        del arg68_1
        del buf56
        # Source Nodes: [out_32], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, arg69_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg69_1
        buf60 = buf59; del buf59  # reuse
        # Source Nodes: [out_33, out_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf60, arg326_1, arg327_1, arg70_1, arg71_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg326_1
        del arg327_1
        del arg70_1
        del arg71_1
        # Source Nodes: [sp_53], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(reinterpret_tensor(buf60, (8, 64, 28, 28), (200704, 784, 28, 1), 0), arg72_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf61, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg72_1
        buf70 = buf54; del buf54  # reuse
        buf62 = reinterpret_tensor(buf70, (8, 64, 28, 28), (200704, 784, 28, 1), 0)  # alias
        buf63 = buf49; del buf49  # reuse
        # Source Nodes: [sp_54, sp_55, sp_56, sp_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15.run(buf61, arg329_1, arg330_1, arg73_1, arg74_1, buf60, buf62, buf63, 401408, grid=grid(401408), stream=stream0)
        del arg329_1
        del arg330_1
        del arg73_1
        del arg74_1
        del buf61
        # Source Nodes: [sp_56, sp_57], Original ATen: [aten.add, aten.convolution]
        buf64 = extern_kernels.convolution(buf63, arg75_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf64, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg75_1
        buf65 = reinterpret_tensor(buf70, (8, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        buf66 = buf63; del buf63  # reuse
        # Source Nodes: [sp_58, sp_59, sp_60, sp_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_16.run(buf64, arg332_1, arg333_1, arg76_1, arg77_1, buf60, buf65, buf66, 401408, grid=grid(401408), stream=stream0)
        del arg332_1
        del arg333_1
        del arg76_1
        del arg77_1
        del buf64
        # Source Nodes: [sp_60, sp_61], Original ATen: [aten.add, aten.convolution]
        buf67 = extern_kernels.convolution(buf66, arg78_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf67, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg78_1
        del buf66
        buf68 = reinterpret_tensor(buf70, (8, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        # Source Nodes: [sp_62, sp_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf67, arg335_1, arg336_1, arg79_1, arg80_1, buf68, 401408, grid=grid(401408), stream=stream0)
        del arg335_1
        del arg336_1
        del arg79_1
        del arg80_1
        buf69 = reinterpret_tensor(buf70, (8, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        # Source Nodes: [cat_27], Original ATen: [aten.cat]
        triton_poi_fused_cat_17.run(buf60, buf69, 401408, grid=grid(401408), stream=stream0)
        del buf60
        del buf62
        del buf65
        del buf68
        del buf69
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg81_1
        buf72 = buf58; del buf58  # reuse
        # Source Nodes: [out_37, out_38, shortcut_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf72, buf71, arg338_1, arg339_1, arg82_1, arg83_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg338_1
        del arg339_1
        del arg82_1
        del arg83_1
        del buf71
        # Source Nodes: [out_40], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg84_1
        buf74 = buf73; del buf73  # reuse
        # Source Nodes: [out_41, out_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf74, arg341_1, arg342_1, arg85_1, arg86_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg341_1
        del arg342_1
        del arg85_1
        del arg86_1
        # Source Nodes: [sp_66], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(reinterpret_tensor(buf74, (8, 64, 28, 28), (200704, 784, 28, 1), 0), arg87_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf75, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg87_1
        buf84 = buf70; del buf70  # reuse
        buf76 = reinterpret_tensor(buf84, (8, 64, 28, 28), (200704, 784, 28, 1), 0)  # alias
        buf77 = buf67; del buf67  # reuse
        # Source Nodes: [sp_67, sp_68, sp_69, sp_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15.run(buf75, arg344_1, arg345_1, arg88_1, arg89_1, buf74, buf76, buf77, 401408, grid=grid(401408), stream=stream0)
        del arg344_1
        del arg345_1
        del arg88_1
        del arg89_1
        del buf75
        # Source Nodes: [sp_69, sp_70], Original ATen: [aten.add, aten.convolution]
        buf78 = extern_kernels.convolution(buf77, arg90_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf78, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg90_1
        buf79 = reinterpret_tensor(buf84, (8, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        buf80 = buf77; del buf77  # reuse
        # Source Nodes: [sp_71, sp_72, sp_73, sp_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_16.run(buf78, arg347_1, arg348_1, arg91_1, arg92_1, buf74, buf79, buf80, 401408, grid=grid(401408), stream=stream0)
        del arg347_1
        del arg348_1
        del arg91_1
        del arg92_1
        del buf78
        # Source Nodes: [sp_73, sp_74], Original ATen: [aten.add, aten.convolution]
        buf81 = extern_kernels.convolution(buf80, arg93_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf81, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg93_1
        del buf80
        buf82 = reinterpret_tensor(buf84, (8, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        # Source Nodes: [sp_75, sp_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf81, arg350_1, arg351_1, arg94_1, arg95_1, buf82, 401408, grid=grid(401408), stream=stream0)
        del arg350_1
        del arg351_1
        del arg94_1
        del arg95_1
        buf83 = reinterpret_tensor(buf84, (8, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        # Source Nodes: [cat_26], Original ATen: [aten.cat]
        triton_poi_fused_cat_17.run(buf74, buf83, 401408, grid=grid(401408), stream=stream0)
        del buf74
        del buf76
        del buf79
        del buf82
        del buf83
        # Source Nodes: [out_44], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg96_1
        buf86 = buf72; del buf72  # reuse
        # Source Nodes: [out_45, out_46, shortcut_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf86, buf85, arg353_1, arg354_1, arg97_1, arg98_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg353_1
        del arg354_1
        del arg97_1
        del arg98_1
        del buf85
        # Source Nodes: [out_48], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg99_1
        buf88 = buf87; del buf87  # reuse
        # Source Nodes: [out_49, out_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf88, arg356_1, arg357_1, arg100_1, arg101_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg100_1
        del arg101_1
        del arg356_1
        del arg357_1
        # Source Nodes: [sp_79], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(reinterpret_tensor(buf88, (8, 64, 28, 28), (200704, 784, 28, 1), 0), arg102_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf89, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg102_1
        buf98 = buf84; del buf84  # reuse
        buf90 = reinterpret_tensor(buf98, (8, 64, 28, 28), (200704, 784, 28, 1), 0)  # alias
        buf91 = buf81; del buf81  # reuse
        # Source Nodes: [sp_80, sp_81, sp_82, sp_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_15.run(buf89, arg359_1, arg360_1, arg103_1, arg104_1, buf88, buf90, buf91, 401408, grid=grid(401408), stream=stream0)
        del arg103_1
        del arg104_1
        del arg359_1
        del arg360_1
        del buf89
        # Source Nodes: [sp_82, sp_83], Original ATen: [aten.add, aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg105_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf92, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg105_1
        buf93 = reinterpret_tensor(buf98, (8, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        buf94 = buf91; del buf91  # reuse
        # Source Nodes: [sp_84, sp_85, sp_86, sp_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_16.run(buf92, arg362_1, arg363_1, arg106_1, arg107_1, buf88, buf93, buf94, 401408, grid=grid(401408), stream=stream0)
        del arg106_1
        del arg107_1
        del arg362_1
        del arg363_1
        del buf92
        # Source Nodes: [sp_86, sp_87], Original ATen: [aten.add, aten.convolution]
        buf95 = extern_kernels.convolution(buf94, arg108_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf95, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg108_1
        del buf94
        buf96 = reinterpret_tensor(buf98, (8, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        # Source Nodes: [sp_88, sp_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf95, arg365_1, arg366_1, arg109_1, arg110_1, buf96, 401408, grid=grid(401408), stream=stream0)
        del arg109_1
        del arg110_1
        del arg365_1
        del arg366_1
        buf97 = reinterpret_tensor(buf98, (8, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        # Source Nodes: [cat_25], Original ATen: [aten.cat]
        triton_poi_fused_cat_17.run(buf88, buf97, 401408, grid=grid(401408), stream=stream0)
        del buf88
        del buf90
        del buf93
        del buf96
        del buf97
        # Source Nodes: [out_52], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg111_1
        del buf98
        buf100 = buf86; del buf86  # reuse
        # Source Nodes: [out_53, out_54, shortcut_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf100, buf99, arg368_1, arg369_1, arg112_1, arg113_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg112_1
        del arg113_1
        del arg368_1
        del arg369_1
        del buf99
        # Source Nodes: [out_56], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg114_1
        buf102 = buf101; del buf101  # reuse
        # Source Nodes: [out_57, out_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf102, arg371_1, arg372_1, arg115_1, arg116_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg115_1
        del arg116_1
        del arg371_1
        del arg372_1
        # Source Nodes: [sp_92], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(reinterpret_tensor(buf102, (8, 128, 28, 28), (401408, 784, 28, 1), 0), arg117_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf103, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg117_1
        # Source Nodes: [sp_96], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(reinterpret_tensor(buf102, (8, 128, 28, 28), (401408, 784, 28, 1), 100352), arg120_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf104, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg120_1
        # Source Nodes: [sp_100], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(reinterpret_tensor(buf102, (8, 128, 28, 28), (401408, 784, 28, 1), 200704), arg123_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf105, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg123_1
        buf110 = reinterpret_tensor(buf39, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf39  # reuse
        buf106 = reinterpret_tensor(buf110, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        # Source Nodes: [getattr_l__mod___layer3___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_20.run(buf102, buf106, 200704, grid=grid(200704), stream=stream0)
        del buf102
        buf107 = reinterpret_tensor(buf110, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
        # Source Nodes: [sp_93, sp_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf103, arg374_1, arg375_1, arg118_1, arg119_1, buf107, 200704, grid=grid(200704), stream=stream0)
        del arg118_1
        del arg119_1
        del arg374_1
        del arg375_1
        del buf103
        buf108 = reinterpret_tensor(buf110, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        # Source Nodes: [sp_97, sp_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf104, arg377_1, arg378_1, arg121_1, arg122_1, buf108, 200704, grid=grid(200704), stream=stream0)
        del arg121_1
        del arg122_1
        del arg377_1
        del arg378_1
        del buf104
        buf109 = reinterpret_tensor(buf110, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        # Source Nodes: [sp_101, sp_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf105, arg380_1, arg381_1, arg124_1, arg125_1, buf109, 200704, grid=grid(200704), stream=stream0)
        del arg124_1
        del arg125_1
        del arg380_1
        del arg381_1
        del buf106
        del buf107
        del buf108
        del buf109
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg126_1
        # Source Nodes: [getattr_l__mod___layer3___0___downsample_0], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf100, arg129_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg129_1
        del buf100
        buf113 = buf111; del buf111  # reuse
        buf114 = buf113; del buf113  # reuse
        # Source Nodes: [out_61, out_62, shortcut_10, shortcut_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf114, arg383_1, arg384_1, arg127_1, arg128_1, buf112, arg386_1, arg387_1, arg130_1, arg131_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg127_1
        del arg128_1
        del arg130_1
        del arg131_1
        del arg383_1
        del arg384_1
        del arg386_1
        del arg387_1
        del buf112
        # Source Nodes: [out_64], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg132_1
        buf116 = buf115; del buf115  # reuse
        # Source Nodes: [out_65, out_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf116, arg389_1, arg390_1, arg133_1, arg134_1, 802816, grid=grid(802816), stream=stream0)
        del arg133_1
        del arg134_1
        del arg389_1
        del arg390_1
        # Source Nodes: [sp_105], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(reinterpret_tensor(buf116, (8, 128, 14, 14), (100352, 196, 14, 1), 0), arg135_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf117, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg135_1
        buf126 = buf110; del buf110  # reuse
        buf118 = reinterpret_tensor(buf126, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
        buf119 = buf105; del buf105  # reuse
        # Source Nodes: [sp_106, sp_107, sp_108, sp_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24.run(buf117, arg392_1, arg393_1, arg136_1, arg137_1, buf116, buf118, buf119, 200704, grid=grid(200704), stream=stream0)
        del arg136_1
        del arg137_1
        del arg392_1
        del arg393_1
        del buf117
        # Source Nodes: [sp_108, sp_109], Original ATen: [aten.add, aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg138_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf120, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg138_1
        buf121 = reinterpret_tensor(buf126, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        buf122 = buf119; del buf119  # reuse
        # Source Nodes: [sp_110, sp_111, sp_112, sp_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_25.run(buf120, arg395_1, arg396_1, arg139_1, arg140_1, buf116, buf121, buf122, 200704, grid=grid(200704), stream=stream0)
        del arg139_1
        del arg140_1
        del arg395_1
        del arg396_1
        del buf120
        # Source Nodes: [sp_112, sp_113], Original ATen: [aten.add, aten.convolution]
        buf123 = extern_kernels.convolution(buf122, arg141_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf123, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg141_1
        del buf122
        buf124 = reinterpret_tensor(buf126, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        # Source Nodes: [sp_114, sp_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf123, arg398_1, arg399_1, arg142_1, arg143_1, buf124, 200704, grid=grid(200704), stream=stream0)
        del arg142_1
        del arg143_1
        del arg398_1
        del arg399_1
        buf125 = reinterpret_tensor(buf126, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_23], Original ATen: [aten.cat]
        triton_poi_fused_cat_26.run(buf116, buf125, 200704, grid=grid(200704), stream=stream0)
        del buf116
        del buf118
        del buf121
        del buf124
        del buf125
        # Source Nodes: [out_68], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg144_1
        buf128 = buf114; del buf114  # reuse
        # Source Nodes: [out_69, out_70, shortcut_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_27.run(buf128, buf127, arg401_1, arg402_1, arg145_1, arg146_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg145_1
        del arg146_1
        del arg401_1
        del arg402_1
        del buf127
        # Source Nodes: [out_72], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg147_1
        buf130 = buf129; del buf129  # reuse
        # Source Nodes: [out_73, out_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf130, arg404_1, arg405_1, arg148_1, arg149_1, 802816, grid=grid(802816), stream=stream0)
        del arg148_1
        del arg149_1
        del arg404_1
        del arg405_1
        # Source Nodes: [sp_118], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(reinterpret_tensor(buf130, (8, 128, 14, 14), (100352, 196, 14, 1), 0), arg150_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf131, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg150_1
        buf140 = buf126; del buf126  # reuse
        buf132 = reinterpret_tensor(buf140, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
        buf133 = buf123; del buf123  # reuse
        # Source Nodes: [sp_119, sp_120, sp_121, sp_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24.run(buf131, arg407_1, arg408_1, arg151_1, arg152_1, buf130, buf132, buf133, 200704, grid=grid(200704), stream=stream0)
        del arg151_1
        del arg152_1
        del arg407_1
        del arg408_1
        del buf131
        # Source Nodes: [sp_121, sp_122], Original ATen: [aten.add, aten.convolution]
        buf134 = extern_kernels.convolution(buf133, arg153_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf134, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg153_1
        buf135 = reinterpret_tensor(buf140, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        buf136 = buf133; del buf133  # reuse
        # Source Nodes: [sp_123, sp_124, sp_125, sp_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_25.run(buf134, arg410_1, arg411_1, arg154_1, arg155_1, buf130, buf135, buf136, 200704, grid=grid(200704), stream=stream0)
        del arg154_1
        del arg155_1
        del arg410_1
        del arg411_1
        del buf134
        # Source Nodes: [sp_125, sp_126], Original ATen: [aten.add, aten.convolution]
        buf137 = extern_kernels.convolution(buf136, arg156_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf137, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg156_1
        del buf136
        buf138 = reinterpret_tensor(buf140, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        # Source Nodes: [sp_127, sp_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf137, arg413_1, arg414_1, arg157_1, arg158_1, buf138, 200704, grid=grid(200704), stream=stream0)
        del arg157_1
        del arg158_1
        del arg413_1
        del arg414_1
        buf139 = reinterpret_tensor(buf140, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_22], Original ATen: [aten.cat]
        triton_poi_fused_cat_26.run(buf130, buf139, 200704, grid=grid(200704), stream=stream0)
        del buf130
        del buf132
        del buf135
        del buf138
        del buf139
        # Source Nodes: [out_76], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg159_1
        buf142 = buf128; del buf128  # reuse
        # Source Nodes: [out_77, out_78, shortcut_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_27.run(buf142, buf141, arg416_1, arg417_1, arg160_1, arg161_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg160_1
        del arg161_1
        del arg416_1
        del arg417_1
        del buf141
        # Source Nodes: [out_80], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg162_1
        buf144 = buf143; del buf143  # reuse
        # Source Nodes: [out_81, out_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf144, arg419_1, arg420_1, arg163_1, arg164_1, 802816, grid=grid(802816), stream=stream0)
        del arg163_1
        del arg164_1
        del arg419_1
        del arg420_1
        # Source Nodes: [sp_131], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(reinterpret_tensor(buf144, (8, 128, 14, 14), (100352, 196, 14, 1), 0), arg165_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf145, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg165_1
        buf154 = buf140; del buf140  # reuse
        buf146 = reinterpret_tensor(buf154, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
        buf147 = buf137; del buf137  # reuse
        # Source Nodes: [sp_132, sp_133, sp_134, sp_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24.run(buf145, arg422_1, arg423_1, arg166_1, arg167_1, buf144, buf146, buf147, 200704, grid=grid(200704), stream=stream0)
        del arg166_1
        del arg167_1
        del arg422_1
        del arg423_1
        del buf145
        # Source Nodes: [sp_134, sp_135], Original ATen: [aten.add, aten.convolution]
        buf148 = extern_kernels.convolution(buf147, arg168_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf148, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg168_1
        buf149 = reinterpret_tensor(buf154, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        buf150 = buf147; del buf147  # reuse
        # Source Nodes: [sp_136, sp_137, sp_138, sp_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_25.run(buf148, arg425_1, arg426_1, arg169_1, arg170_1, buf144, buf149, buf150, 200704, grid=grid(200704), stream=stream0)
        del arg169_1
        del arg170_1
        del arg425_1
        del arg426_1
        del buf148
        # Source Nodes: [sp_138, sp_139], Original ATen: [aten.add, aten.convolution]
        buf151 = extern_kernels.convolution(buf150, arg171_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf151, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg171_1
        del buf150
        buf152 = reinterpret_tensor(buf154, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        # Source Nodes: [sp_140, sp_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf151, arg428_1, arg429_1, arg172_1, arg173_1, buf152, 200704, grid=grid(200704), stream=stream0)
        del arg172_1
        del arg173_1
        del arg428_1
        del arg429_1
        buf153 = reinterpret_tensor(buf154, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_21], Original ATen: [aten.cat]
        triton_poi_fused_cat_26.run(buf144, buf153, 200704, grid=grid(200704), stream=stream0)
        del buf144
        del buf146
        del buf149
        del buf152
        del buf153
        # Source Nodes: [out_84], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf154, arg174_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg174_1
        buf156 = buf142; del buf142  # reuse
        # Source Nodes: [out_85, out_86, shortcut_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_27.run(buf156, buf155, arg431_1, arg432_1, arg175_1, arg176_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg175_1
        del arg176_1
        del arg431_1
        del arg432_1
        del buf155
        # Source Nodes: [out_88], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, arg177_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg177_1
        buf158 = buf157; del buf157  # reuse
        # Source Nodes: [out_89, out_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf158, arg434_1, arg435_1, arg178_1, arg179_1, 802816, grid=grid(802816), stream=stream0)
        del arg178_1
        del arg179_1
        del arg434_1
        del arg435_1
        # Source Nodes: [sp_144], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(reinterpret_tensor(buf158, (8, 128, 14, 14), (100352, 196, 14, 1), 0), arg180_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf159, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg180_1
        buf168 = buf154; del buf154  # reuse
        buf160 = reinterpret_tensor(buf168, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
        buf161 = buf151; del buf151  # reuse
        # Source Nodes: [sp_145, sp_146, sp_147, sp_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24.run(buf159, arg437_1, arg438_1, arg181_1, arg182_1, buf158, buf160, buf161, 200704, grid=grid(200704), stream=stream0)
        del arg181_1
        del arg182_1
        del arg437_1
        del arg438_1
        del buf159
        # Source Nodes: [sp_147, sp_148], Original ATen: [aten.add, aten.convolution]
        buf162 = extern_kernels.convolution(buf161, arg183_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf162, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg183_1
        buf163 = reinterpret_tensor(buf168, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        buf164 = buf161; del buf161  # reuse
        # Source Nodes: [sp_149, sp_150, sp_151, sp_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_25.run(buf162, arg440_1, arg441_1, arg184_1, arg185_1, buf158, buf163, buf164, 200704, grid=grid(200704), stream=stream0)
        del arg184_1
        del arg185_1
        del arg440_1
        del arg441_1
        del buf162
        # Source Nodes: [sp_151, sp_152], Original ATen: [aten.add, aten.convolution]
        buf165 = extern_kernels.convolution(buf164, arg186_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf165, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg186_1
        del buf164
        buf166 = reinterpret_tensor(buf168, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        # Source Nodes: [sp_153, sp_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf165, arg443_1, arg444_1, arg187_1, arg188_1, buf166, 200704, grid=grid(200704), stream=stream0)
        del arg187_1
        del arg188_1
        del arg443_1
        del arg444_1
        buf167 = reinterpret_tensor(buf168, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_20], Original ATen: [aten.cat]
        triton_poi_fused_cat_26.run(buf158, buf167, 200704, grid=grid(200704), stream=stream0)
        del buf158
        del buf160
        del buf163
        del buf166
        del buf167
        # Source Nodes: [out_92], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, arg189_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg189_1
        buf170 = buf156; del buf156  # reuse
        # Source Nodes: [out_93, out_94, shortcut_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_27.run(buf170, buf169, arg446_1, arg447_1, arg190_1, arg191_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg190_1
        del arg191_1
        del arg446_1
        del arg447_1
        del buf169
        # Source Nodes: [out_96], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, arg192_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg192_1
        buf172 = buf171; del buf171  # reuse
        # Source Nodes: [out_97, out_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf172, arg449_1, arg450_1, arg193_1, arg194_1, 802816, grid=grid(802816), stream=stream0)
        del arg193_1
        del arg194_1
        del arg449_1
        del arg450_1
        # Source Nodes: [sp_157], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(reinterpret_tensor(buf172, (8, 128, 14, 14), (100352, 196, 14, 1), 0), arg195_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf173, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg195_1
        buf182 = buf168; del buf168  # reuse
        buf174 = reinterpret_tensor(buf182, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
        buf175 = buf165; del buf165  # reuse
        # Source Nodes: [sp_158, sp_159, sp_160, sp_161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_24.run(buf173, arg452_1, arg453_1, arg196_1, arg197_1, buf172, buf174, buf175, 200704, grid=grid(200704), stream=stream0)
        del arg196_1
        del arg197_1
        del arg452_1
        del arg453_1
        del buf173
        # Source Nodes: [sp_160, sp_161], Original ATen: [aten.add, aten.convolution]
        buf176 = extern_kernels.convolution(buf175, arg198_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf176, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg198_1
        buf177 = reinterpret_tensor(buf182, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        buf178 = buf175; del buf175  # reuse
        # Source Nodes: [sp_162, sp_163, sp_164, sp_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_25.run(buf176, arg455_1, arg456_1, arg199_1, arg200_1, buf172, buf177, buf178, 200704, grid=grid(200704), stream=stream0)
        del arg199_1
        del arg200_1
        del arg455_1
        del arg456_1
        del buf176
        # Source Nodes: [sp_164, sp_165], Original ATen: [aten.add, aten.convolution]
        buf179 = extern_kernels.convolution(buf178, arg201_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf179, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg201_1
        del buf178
        buf180 = reinterpret_tensor(buf182, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        # Source Nodes: [sp_166, sp_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf179, arg458_1, arg459_1, arg202_1, arg203_1, buf180, 200704, grid=grid(200704), stream=stream0)
        del arg202_1
        del arg203_1
        del arg458_1
        del arg459_1
        del buf179
        buf181 = reinterpret_tensor(buf182, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_19], Original ATen: [aten.cat]
        triton_poi_fused_cat_26.run(buf172, buf181, 200704, grid=grid(200704), stream=stream0)
        del buf172
        del buf174
        del buf177
        del buf180
        del buf181
        # Source Nodes: [out_100], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, arg204_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg204_1
        del buf182
        buf184 = buf170; del buf170  # reuse
        # Source Nodes: [out_101, out_102, shortcut_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_27.run(buf184, buf183, arg461_1, arg462_1, arg205_1, arg206_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg205_1
        del arg206_1
        del arg461_1
        del arg462_1
        del buf183
        # Source Nodes: [out_104], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg207_1
        buf186 = buf185; del buf185  # reuse
        # Source Nodes: [out_105, out_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_28.run(buf186, arg464_1, arg465_1, arg208_1, arg209_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg208_1
        del arg209_1
        del arg464_1
        del arg465_1
        # Source Nodes: [sp_170], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(reinterpret_tensor(buf186, (8, 256, 14, 14), (200704, 196, 14, 1), 0), arg210_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf187, (8, 256, 7, 7), (12544, 49, 7, 1))
        del arg210_1
        # Source Nodes: [sp_174], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(reinterpret_tensor(buf186, (8, 256, 14, 14), (200704, 196, 14, 1), 50176), arg213_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf188, (8, 256, 7, 7), (12544, 49, 7, 1))
        del arg213_1
        # Source Nodes: [sp_178], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(reinterpret_tensor(buf186, (8, 256, 14, 14), (200704, 196, 14, 1), 100352), arg216_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf189, (8, 256, 7, 7), (12544, 49, 7, 1))
        del arg216_1
        buf194 = reinterpret_tensor(buf95, (8, 1024, 7, 7), (50176, 49, 7, 1), 0); del buf95  # reuse
        buf190 = reinterpret_tensor(buf194, (8, 256, 7, 7), (50176, 49, 7, 1), 37632)  # alias
        # Source Nodes: [getattr_l__mod___layer4___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_29.run(buf186, buf190, 100352, grid=grid(100352), stream=stream0)
        del buf186
        buf191 = reinterpret_tensor(buf194, (8, 256, 7, 7), (50176, 49, 7, 1), 0)  # alias
        # Source Nodes: [sp_171, sp_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf187, arg467_1, arg468_1, arg211_1, arg212_1, buf191, 100352, grid=grid(100352), stream=stream0)
        del arg211_1
        del arg212_1
        del arg467_1
        del arg468_1
        del buf187
        buf192 = reinterpret_tensor(buf194, (8, 256, 7, 7), (50176, 49, 7, 1), 12544)  # alias
        # Source Nodes: [sp_175, sp_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf188, arg470_1, arg471_1, arg214_1, arg215_1, buf192, 100352, grid=grid(100352), stream=stream0)
        del arg214_1
        del arg215_1
        del arg470_1
        del arg471_1
        del buf188
        buf193 = reinterpret_tensor(buf194, (8, 256, 7, 7), (50176, 49, 7, 1), 25088)  # alias
        # Source Nodes: [sp_179, sp_180], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf189, arg473_1, arg474_1, arg217_1, arg218_1, buf193, 100352, grid=grid(100352), stream=stream0)
        del arg217_1
        del arg218_1
        del arg473_1
        del arg474_1
        del buf190
        del buf191
        del buf192
        del buf193
        # Source Nodes: [out_108], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf194, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (8, 2048, 7, 7), (100352, 49, 7, 1))
        del arg219_1
        # Source Nodes: [getattr_l__mod___layer4___0___downsample_0], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf184, arg222_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (8, 2048, 7, 7), (100352, 49, 7, 1))
        del arg222_1
        del buf184
        buf197 = buf195; del buf195  # reuse
        buf198 = buf197; del buf197  # reuse
        # Source Nodes: [out_109, out_110, shortcut_17, shortcut_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31.run(buf198, arg476_1, arg477_1, arg220_1, arg221_1, buf196, arg479_1, arg480_1, arg223_1, arg224_1, 802816, grid=grid(802816), stream=stream0)
        del arg220_1
        del arg221_1
        del arg223_1
        del arg224_1
        del arg476_1
        del arg477_1
        del arg479_1
        del arg480_1
        del buf196
        # Source Nodes: [out_112], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (8, 1024, 7, 7), (50176, 49, 7, 1))
        del arg225_1
        buf200 = buf199; del buf199  # reuse
        # Source Nodes: [out_113, out_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf200, arg482_1, arg483_1, arg226_1, arg227_1, 401408, grid=grid(401408), stream=stream0)
        del arg226_1
        del arg227_1
        del arg482_1
        del arg483_1
        # Source Nodes: [sp_183], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(reinterpret_tensor(buf200, (8, 256, 7, 7), (50176, 49, 7, 1), 0), arg228_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf201, (8, 256, 7, 7), (12544, 49, 7, 1))
        del arg228_1
        buf210 = buf194; del buf194  # reuse
        buf202 = reinterpret_tensor(buf210, (8, 256, 7, 7), (50176, 49, 7, 1), 0)  # alias
        buf203 = buf189; del buf189  # reuse
        # Source Nodes: [sp_184, sp_185, sp_186, sp_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_33.run(buf201, arg485_1, arg486_1, arg229_1, arg230_1, buf200, buf202, buf203, 100352, grid=grid(100352), stream=stream0)
        del arg229_1
        del arg230_1
        del arg485_1
        del arg486_1
        del buf201
        # Source Nodes: [sp_186, sp_187], Original ATen: [aten.add, aten.convolution]
        buf204 = extern_kernels.convolution(buf203, arg231_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf204, (8, 256, 7, 7), (12544, 49, 7, 1))
        del arg231_1
        buf205 = reinterpret_tensor(buf210, (8, 256, 7, 7), (50176, 49, 7, 1), 12544)  # alias
        buf206 = buf203; del buf203  # reuse
        # Source Nodes: [sp_188, sp_189, sp_190, sp_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_34.run(buf204, arg488_1, arg489_1, arg232_1, arg233_1, buf200, buf205, buf206, 100352, grid=grid(100352), stream=stream0)
        del arg232_1
        del arg233_1
        del arg488_1
        del arg489_1
        del buf204
        # Source Nodes: [sp_190, sp_191], Original ATen: [aten.add, aten.convolution]
        buf207 = extern_kernels.convolution(buf206, arg234_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf207, (8, 256, 7, 7), (12544, 49, 7, 1))
        del arg234_1
        del buf206
        buf208 = reinterpret_tensor(buf210, (8, 256, 7, 7), (50176, 49, 7, 1), 25088)  # alias
        # Source Nodes: [sp_192, sp_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf207, arg491_1, arg492_1, arg235_1, arg236_1, buf208, 100352, grid=grid(100352), stream=stream0)
        del arg235_1
        del arg236_1
        del arg491_1
        del arg492_1
        buf209 = reinterpret_tensor(buf210, (8, 256, 7, 7), (50176, 49, 7, 1), 37632)  # alias
        # Source Nodes: [cat_17], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf200, buf209, 100352, grid=grid(100352), stream=stream0)
        del buf200
        del buf202
        del buf205
        del buf208
        del buf209
        # Source Nodes: [out_116], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, arg237_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (8, 2048, 7, 7), (100352, 49, 7, 1))
        del arg237_1
        buf212 = buf198; del buf198  # reuse
        # Source Nodes: [out_117, out_118, shortcut_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf212, buf211, arg494_1, arg495_1, arg238_1, arg239_1, 802816, grid=grid(802816), stream=stream0)
        del arg238_1
        del arg239_1
        del arg494_1
        del arg495_1
        del buf211
        # Source Nodes: [out_120], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, arg240_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (8, 1024, 7, 7), (50176, 49, 7, 1))
        del arg240_1
        buf214 = buf213; del buf213  # reuse
        # Source Nodes: [out_121, out_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf214, arg497_1, arg498_1, arg241_1, arg242_1, 401408, grid=grid(401408), stream=stream0)
        del arg241_1
        del arg242_1
        del arg497_1
        del arg498_1
        # Source Nodes: [sp_196], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(reinterpret_tensor(buf214, (8, 256, 7, 7), (50176, 49, 7, 1), 0), arg243_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf215, (8, 256, 7, 7), (12544, 49, 7, 1))
        del arg243_1
        buf224 = buf210; del buf210  # reuse
        buf216 = reinterpret_tensor(buf224, (8, 256, 7, 7), (50176, 49, 7, 1), 0)  # alias
        buf217 = buf207; del buf207  # reuse
        # Source Nodes: [sp_197, sp_198, sp_199, sp_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_33.run(buf215, arg500_1, arg501_1, arg244_1, arg245_1, buf214, buf216, buf217, 100352, grid=grid(100352), stream=stream0)
        del arg244_1
        del arg245_1
        del arg500_1
        del arg501_1
        del buf215
        # Source Nodes: [sp_199, sp_200], Original ATen: [aten.add, aten.convolution]
        buf218 = extern_kernels.convolution(buf217, arg246_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf218, (8, 256, 7, 7), (12544, 49, 7, 1))
        del arg246_1
        buf219 = reinterpret_tensor(buf224, (8, 256, 7, 7), (50176, 49, 7, 1), 12544)  # alias
        buf220 = buf217; del buf217  # reuse
        # Source Nodes: [sp_201, sp_202, sp_203, sp_204], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_34.run(buf218, arg503_1, arg504_1, arg247_1, arg248_1, buf214, buf219, buf220, 100352, grid=grid(100352), stream=stream0)
        del arg247_1
        del arg248_1
        del arg503_1
        del arg504_1
        del buf218
        # Source Nodes: [sp_203, sp_204], Original ATen: [aten.add, aten.convolution]
        buf221 = extern_kernels.convolution(buf220, arg249_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf221, (8, 256, 7, 7), (12544, 49, 7, 1))
        del arg249_1
        del buf220
        buf222 = reinterpret_tensor(buf224, (8, 256, 7, 7), (50176, 49, 7, 1), 25088)  # alias
        # Source Nodes: [sp_205, sp_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_30.run(buf221, arg506_1, arg507_1, arg250_1, arg251_1, buf222, 100352, grid=grid(100352), stream=stream0)
        del arg250_1
        del arg251_1
        del arg506_1
        del arg507_1
        del buf221
        buf223 = reinterpret_tensor(buf224, (8, 256, 7, 7), (50176, 49, 7, 1), 37632)  # alias
        # Source Nodes: [cat_16], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf214, buf223, 100352, grid=grid(100352), stream=stream0)
        del buf214
        del buf216
        del buf219
        del buf222
        del buf223
        # Source Nodes: [out_124], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, arg252_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (8, 2048, 7, 7), (100352, 49, 7, 1))
        del arg252_1
        del buf224
        buf226 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cuda', dtype=torch.float32)
        buf227 = reinterpret_tensor(buf226, (8, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf226  # reuse
        # Source Nodes: [out_125, out_126, x_8, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_37.run(buf227, buf225, arg509_1, arg510_1, arg253_1, arg254_1, buf212, 16384, 49, grid=grid(16384), stream=stream0)
        del arg253_1
        del arg254_1
        del arg509_1
        del arg510_1
        del buf212
        del buf225
        buf228 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg256_1, reinterpret_tensor(buf227, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg255_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf228)
        del arg255_1
        del arg256_1
        return (buf228, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg260_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg263_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg266_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg269_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg272_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg275_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg278_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg281_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg284_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg287_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg290_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg293_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg296_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg299_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg302_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg305_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg308_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg311_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg314_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg317_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg320_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg323_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg326_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg329_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg332_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg335_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg338_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg341_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg344_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg347_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg350_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg353_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg356_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg359_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg362_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg365_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg368_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg371_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg374_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg377_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg380_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg383_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg386_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg389_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg392_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg395_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg398_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg401_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg404_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg407_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg410_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg413_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg416_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg419_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg422_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg425_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg428_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg431_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg434_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg437_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg440_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg443_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg446_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg449_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg452_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg455_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg458_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg461_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg464_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg467_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg470_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg473_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg476_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg479_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg482_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg485_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg488_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg491_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg494_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg497_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg500_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg503_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg506_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg509_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg512_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('res2next50', benchmark_compiled_module)
