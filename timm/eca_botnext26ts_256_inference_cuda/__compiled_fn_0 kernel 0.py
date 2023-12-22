
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


# kernel path: /tmp/torchinductor_youkaichao/as/casi6eir7llhxqtkdzenr6bu6qdc4hyqikzczvlhok3vn6qyt3ng.py
# Source Nodes: [x_1, x_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_1 => add_1, mul_1, mul_2, sub
# x_4 => mul_3, sigmoid
# x_5 => convolution_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 24
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/aw/cawmsuwu7eb7r4k5mxiw53rsiv2aw3par24vmneawccm7zpxo3ih.py
# Source Nodes: [x_10, x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_10 => convolution_2
# x_6 => add_3, mul_5, mul_6, sub_1
# x_9 => mul_7, sigmoid_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 32
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ee/ceeslzc2dvwpqyglcli7qduwx56qinhqezzznxjjbo7beonjabnb.py
# Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_11 => add_5, mul_10, mul_9, sub_2
triton_poi_fused__native_batch_norm_legit_no_training_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 64
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
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/an/cany723znzod2v3nurmntt4op43bo5bfqrrpidsghxgevk6zmkc2.py
# Source Nodes: [shortcut, x_14], Original ATen: [aten.max_pool2d_with_indices, aten.silu]
# shortcut => max_pool2d_with_indices
# x_14 => mul_11, sigmoid_2
triton_poi_fused_max_pool2d_with_indices_silu_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_silu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64) % 64
    x0 = xindex % 64
    x3 = (xindex // 64)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-129) + (2*x0) + (256*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, float("-inf"), tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = 2*x0
    tmp17 = tmp16 >= tmp1
    tmp18 = tmp16 < tmp3
    tmp19 = tmp17 & tmp18
    tmp20 = tmp5 & tmp19
    tmp21 = tl.load(in_ptr0 + ((-128) + (2*x0) + (256*x3)), tmp20, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.full(tmp23.shape, float("-inf"), tmp23.dtype)
    tmp25 = tl.where(tmp20, tmp23, tmp24)
    tmp26 = triton_helpers.maximum(tmp25, tmp15)
    tmp27 = 1 + (2*x0)
    tmp28 = tmp27 >= tmp1
    tmp29 = tmp27 < tmp3
    tmp30 = tmp28 & tmp29
    tmp31 = tmp5 & tmp30
    tmp32 = tl.load(in_ptr0 + ((-127) + (2*x0) + (256*x3)), tmp31, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.sigmoid(tmp32)
    tmp34 = tmp32 * tmp33
    tmp35 = tl.full(tmp34.shape, float("-inf"), tmp34.dtype)
    tmp36 = tl.where(tmp31, tmp34, tmp35)
    tmp37 = triton_helpers.maximum(tmp36, tmp26)
    tmp38 = 2*x1
    tmp39 = tmp38 >= tmp1
    tmp40 = tmp38 < tmp3
    tmp41 = tmp39 & tmp40
    tmp42 = tmp41 & tmp9
    tmp43 = tl.load(in_ptr0 + ((-1) + (2*x0) + (256*x3)), tmp42, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.sigmoid(tmp43)
    tmp45 = tmp43 * tmp44
    tmp46 = tl.full(tmp45.shape, float("-inf"), tmp45.dtype)
    tmp47 = tl.where(tmp42, tmp45, tmp46)
    tmp48 = triton_helpers.maximum(tmp47, tmp37)
    tmp49 = tmp41 & tmp19
    tmp50 = tl.load(in_ptr0 + ((2*x0) + (256*x3)), tmp49, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.sigmoid(tmp50)
    tmp52 = tmp50 * tmp51
    tmp53 = tl.full(tmp52.shape, float("-inf"), tmp52.dtype)
    tmp54 = tl.where(tmp49, tmp52, tmp53)
    tmp55 = triton_helpers.maximum(tmp54, tmp48)
    tmp56 = tmp41 & tmp30
    tmp57 = tl.load(in_ptr0 + (1 + (2*x0) + (256*x3)), tmp56, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.sigmoid(tmp57)
    tmp59 = tmp57 * tmp58
    tmp60 = tl.full(tmp59.shape, float("-inf"), tmp59.dtype)
    tmp61 = tl.where(tmp56, tmp59, tmp60)
    tmp62 = triton_helpers.maximum(tmp61, tmp55)
    tmp63 = 1 + (2*x1)
    tmp64 = tmp63 >= tmp1
    tmp65 = tmp63 < tmp3
    tmp66 = tmp64 & tmp65
    tmp67 = tmp66 & tmp9
    tmp68 = tl.load(in_ptr0 + (127 + (2*x0) + (256*x3)), tmp67, eviction_policy='evict_last', other=0.0)
    tmp69 = tl.sigmoid(tmp68)
    tmp70 = tmp68 * tmp69
    tmp71 = tl.full(tmp70.shape, float("-inf"), tmp70.dtype)
    tmp72 = tl.where(tmp67, tmp70, tmp71)
    tmp73 = triton_helpers.maximum(tmp72, tmp62)
    tmp74 = tmp66 & tmp19
    tmp75 = tl.load(in_ptr0 + (128 + (2*x0) + (256*x3)), tmp74, eviction_policy='evict_last', other=0.0)
    tmp76 = tl.sigmoid(tmp75)
    tmp77 = tmp75 * tmp76
    tmp78 = tl.full(tmp77.shape, float("-inf"), tmp77.dtype)
    tmp79 = tl.where(tmp74, tmp77, tmp78)
    tmp80 = triton_helpers.maximum(tmp79, tmp73)
    tmp81 = tmp66 & tmp30
    tmp82 = tl.load(in_ptr0 + (129 + (2*x0) + (256*x3)), tmp81, eviction_policy='evict_last', other=0.0)
    tmp83 = tl.sigmoid(tmp82)
    tmp84 = tmp82 * tmp83
    tmp85 = tl.full(tmp84.shape, float("-inf"), tmp84.dtype)
    tmp86 = tl.where(tmp81, tmp84, tmp85)
    tmp87 = triton_helpers.maximum(tmp86, tmp80)
    tl.store(out_ptr0 + (x4), tmp87, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/c4/cc4kbpl6v7f33aq7lqlhv3ix3udspqvreuept35aimevfe7mxmtr.py
# Source Nodes: [x_17, x_21, x_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_17 => add_7, mul_13, mul_14, sub_3
# x_21 => mul_15, sigmoid_3
# x_22 => convolution_4
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 64
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7o/c7oxzbajoz3vlkjabgdpz6giukewk24tpdz2rxwmwf6qhuvkl6gs.py
# Source Nodes: [mean, x_23, x_27, y_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.silu]
# mean => mean
# x_23 => add_9, mul_17, mul_18, sub_4
# x_27 => mul_19, sigmoid_4
# y_1 => convolution_5
triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_silu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_silu_5', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 64
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + (4096*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
        tmp15 = tl.sigmoid(tmp14)
        tmp16 = tmp14 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tl.store(in_out_ptr0 + (r2 + (4096*x3)), tmp14, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp20 = 4096.0
    tmp21 = tmp18 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mf/cmfizhm2kq2awqxtbogi6sc6274oimi2xkwu2se6y5ck3ac3k64f.py
# Source Nodes: [x_27, x_29, x_30], Original ATen: [aten.convolution, aten.mul, aten.silu]
# x_27 => mul_19, sigmoid_4
# x_29 => mul_20
# x_30 => convolution_6
triton_poi_fused_convolution_mul_silu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_silu_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 4096)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x2), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5i/c5iiuo5ey2apcjleh5ct2jplaryp5hvlldfmso3atvnucqoo5xty.py
# Source Nodes: [shortcut_1, x_31, x_39, x_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# shortcut_1 => mul_27, sigmoid_6
# x_31 => add_11, mul_22, mul_23, sub_5
# x_39 => add_13, mul_25, mul_26, sub_6
# x_43 => add_14
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 256
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
    tmp29 = tl.sigmoid(tmp28)
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/x4/cx4rlzj6tgknfs6j5uixn3ausn6qf266nwnw7lt5tlxefkuc4ssg.py
# Source Nodes: [shortcut_2, x_59, x_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# shortcut_2 => mul_40, sigmoid_10
# x_59 => add_20, mul_38, mul_39, sub_9
# x_66 => add_21
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 256
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
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tl.store(in_out_ptr0 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dm/cdmfdxisrzxgnidvqiihsekdigovsxpkc2i2lv7jggzoi7x3ozwe.py
# Source Nodes: [x_68, x_72, x_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_68 => add_23, mul_42, mul_43, sub_10
# x_72 => mul_44, sigmoid_11
# x_73 => convolution_13
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 128
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zt/cztnkrokaczwvhuypvfe6ifnldpj5i5ky4wrzh7oqhkszgz3jutl.py
# Source Nodes: [mean_2, x_74, x_78, y_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.silu]
# mean_2 => mean_2
# x_74 => add_25, mul_46, mul_47, sub_11
# x_78 => mul_48, sigmoid_12
# y_7 => convolution_14
triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_silu_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_silu_10', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (r2 + (1024*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 1024.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (1024*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3w/c3wy6ez7fj5u66xy5pog5bdawttly7rba77gflppht2l3cyenqgm.py
# Source Nodes: [x_78, x_80, x_81], Original ATen: [aten.convolution, aten.mul, aten.silu]
# x_78 => mul_48, sigmoid_12
# x_80 => mul_49
# x_81 => convolution_15
triton_poi_fused_convolution_mul_silu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_silu_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x2), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4l/c4lybr7t3uuhkwv653tkrm7jfnbfimt5i4nusxbfaw6vto4zim3l.py
# Source Nodes: [shortcut_3, x_82, x_90, x_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# shortcut_3 => mul_56, sigmoid_14
# x_82 => add_27, mul_51, mul_52, sub_12
# x_90 => add_29, mul_54, mul_55, sub_13
# x_94 => add_30
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 512
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
    tmp29 = tl.sigmoid(tmp28)
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nv/cnv4tpwb5k5tm7arzc4ttizsmlep3ymfnxolae3ejy66pwzy6vxz.py
# Source Nodes: [x_100, x_101, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_100 => mul_60, sigmoid_15
# x_101 => convolution_18
# x_96 => add_32, mul_58, mul_59, sub_14
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 128
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vx/cvxvimpnyifmxt23bhoqa6yymuqevk5tnuly6ltqybdtggccogrk.py
# Source Nodes: [shortcut_4, x_110, x_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# shortcut_4 => mul_69, sigmoid_18
# x_110 => add_36, mul_67, mul_68, sub_16
# x_117 => add_37
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 512
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
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tl.store(in_out_ptr0 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rx/crxzjospvmb7aazfxpmfxynkemxwve3pnw7xoekeid2mnf2ws6mb.py
# Source Nodes: [x_119, x_123, x_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_119 => add_39, mul_71, mul_72, sub_17
# x_123 => mul_73, sigmoid_19
# x_124 => convolution_22
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 256
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7r/c7rjpf526o62ccunvpdm2dtzjlc6wcfgbz5bggy5gfkjzrnpplnd.py
# Source Nodes: [mean_4, x_125, x_129, y_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.silu]
# mean_4 => mean_4
# x_125 => add_41, mul_75, mul_76, sub_18
# x_129 => mul_77, sigmoid_20
# y_13 => convolution_23
triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_silu_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_silu_16', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (r2 + (256*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 256.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (256*x3)), tmp14, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kn/cknimkurjmthsgidgk2jbcuospphkbql222efntsmag3uafgptou.py
# Source Nodes: [x_129, x_131, x_132], Original ATen: [aten.convolution, aten.mul, aten.silu]
# x_129 => mul_77, sigmoid_20
# x_131 => mul_78
# x_132 => convolution_24
triton_poi_fused_convolution_mul_silu_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_silu_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x2), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gr/cgrcxqndwcs6wmvv4vyza4nep4p3xxdccdo27kc427bi4cqv5tdb.py
# Source Nodes: [shortcut_5, x_133, x_141, x_145], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# shortcut_5 => mul_85, sigmoid_22
# x_133 => add_43, mul_80, mul_81, sub_19
# x_141 => add_45, mul_83, mul_84, sub_20
# x_145 => add_46
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1024
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
    tmp29 = tl.sigmoid(tmp28)
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pa/cpaqrjksyz6yeighqhuhqqh6ugmeccvbvfobly6shvgf7e6ij3yq.py
# Source Nodes: [x_147, x_151, x_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_147 => add_48, mul_87, mul_88, sub_21
# x_151 => mul_89, sigmoid_23
# x_153 => convolution_27
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 256
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/su/csuweiu72gjeo3gleglweskjil7ovvn7nbaj4zyveh2rpaxrk73g.py
# Source Nodes: [reshape], Original ATen: [aten.clone]
# reshape => clone
triton_poi_fused_clone_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16384
    x1 = (xindex // 16384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (98304*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xu/cxuzku5sbqv5x7a6b3bkh3bicyzglmhi3jrz6gxllo5u7eituyn7.py
# Source Nodes: [k_1], Original ATen: [aten.clone]
# k_1 => clone_1
triton_poi_fused_clone_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16384
    x1 = (xindex // 16384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (16384 + x0 + (98304*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ui/cui4ocob5em2fzw44xjwohzuzc5f5sgz2rsoebrskxixnugvydhu.py
# Source Nodes: [x_158], Original ATen: [aten.clone]
# x_158 => clone_4
triton_poi_fused_clone_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 16
    x3 = (xindex // 16)
    y0 = yindex % 16
    y1 = (yindex // 16)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (16*x3) + (256*x2) + (256*((y0 + (16*x3)) // 256)) + (4096*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x5 + (256*y4)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/co/ccoi6sczpift3vttnmtj4mfp3boxssmvrfek43a62oevxeaxupi3.py
# Source Nodes: [x_154], Original ATen: [aten.clone]
# x_154 => clone_3
triton_poi_fused_clone_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (4096*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (16*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qw/cqwn26jttnhhxf5ffdebhszvtxfrpfdhlvxrp2ksget3c7evakrv.py
# Source Nodes: [attn, attn_1, mul_5], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn => add_50
# attn_1 => amax, div, exp, sub_22, sum_1
# mul_5 => mul_90
triton_red_fused__softmax_add_mul_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_mul_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp28 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.25
        tmp2 = tmp0 * tmp1
        tmp3 = 15 + (31*(x0 // 16)) + (r2 // 16)
        tmp4 = tl.full([1, 1], 512, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
        tmp7 = tl.full([1, 1], 31, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp5
        tmp10 = tl.load(in_ptr1 + ((31*((15 + (31*(x0 // 16)) + (r2 // 16)) // 32)) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp9, tmp10, tmp11)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp5, tmp12, tmp13)
        tmp15 = 15 + (31*(x0 % 16)) + (r2 % 16)
        tmp16 = tmp15 < tmp4
        tmp17 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
        tmp18 = tmp17 < tmp7
        tmp19 = tmp18 & tmp16
        tmp20 = tl.load(in_ptr2 + ((31*(((15 + (31*(x0 % 16)) + (r2 % 16)) // 32) % 16)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), rmask & tmp19, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
        tmp22 = tl.where(tmp19, tmp20, tmp21)
        tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
        tmp24 = tl.where(tmp16, tmp22, tmp23)
        tmp25 = tmp14 + tmp24
        tmp26 = tmp2 + tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = triton_helpers.maximum(_tmp28, tmp27)
        _tmp28 = tl.where(rmask, tmp29, _tmp28)
    tmp28 = triton_helpers.max2(_tmp28, 1)[:, None]
    _tmp60 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp30 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = 0.25
        tmp32 = tmp30 * tmp31
        tmp33 = 15 + (31*(x0 // 16)) + (r2 // 16)
        tmp34 = tl.full([1, 1], 512, tl.int64)
        tmp35 = tmp33 < tmp34
        tmp36 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
        tmp37 = tl.full([1, 1], 31, tl.int64)
        tmp38 = tmp36 < tmp37
        tmp39 = tmp38 & tmp35
        tmp40 = tl.load(in_ptr1 + ((31*((15 + (31*(x0 // 16)) + (r2 // 16)) // 32)) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), rmask & tmp39, eviction_policy='evict_last', other=0.0)
        tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
        tmp42 = tl.where(tmp39, tmp40, tmp41)
        tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
        tmp44 = tl.where(tmp35, tmp42, tmp43)
        tmp45 = 15 + (31*(x0 % 16)) + (r2 % 16)
        tmp46 = tmp45 < tmp34
        tmp47 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
        tmp48 = tmp47 < tmp37
        tmp49 = tmp48 & tmp46
        tmp50 = tl.load(in_ptr2 + ((31*(((15 + (31*(x0 % 16)) + (r2 % 16)) // 32) % 16)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), rmask & tmp49, eviction_policy='evict_last', other=0.0)
        tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
        tmp52 = tl.where(tmp49, tmp50, tmp51)
        tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
        tmp54 = tl.where(tmp46, tmp52, tmp53)
        tmp55 = tmp44 + tmp54
        tmp56 = tmp32 + tmp55
        tmp57 = tmp56 - tmp28
        tmp58 = tl.exp(tmp57)
        tmp59 = tl.broadcast_to(tmp58, [XBLOCK, RBLOCK])
        tmp61 = _tmp60 + tmp59
        _tmp60 = tl.where(rmask, tmp61, _tmp60)
    tmp60 = tl.sum(_tmp60, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp62 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp63 = 0.25
        tmp64 = tmp62 * tmp63
        tmp65 = 15 + (31*(x0 // 16)) + (r2 // 16)
        tmp66 = tl.full([1, 1], 512, tl.int64)
        tmp67 = tmp65 < tmp66
        tmp68 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
        tmp69 = tl.full([1, 1], 31, tl.int64)
        tmp70 = tmp68 < tmp69
        tmp71 = tmp70 & tmp67
        tmp72 = tl.load(in_ptr1 + ((31*((15 + (31*(x0 // 16)) + (r2 // 16)) // 32)) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), rmask & tmp71, eviction_policy='evict_last', other=0.0)
        tmp73 = tl.full(tmp72.shape, 0.0, tmp72.dtype)
        tmp74 = tl.where(tmp71, tmp72, tmp73)
        tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
        tmp76 = tl.where(tmp67, tmp74, tmp75)
        tmp77 = 15 + (31*(x0 % 16)) + (r2 % 16)
        tmp78 = tmp77 < tmp66
        tmp79 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
        tmp80 = tmp79 < tmp69
        tmp81 = tmp80 & tmp78
        tmp82 = tl.load(in_ptr2 + ((31*(((15 + (31*(x0 % 16)) + (r2 % 16)) // 32) % 16)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), rmask & tmp81, eviction_policy='evict_last', other=0.0)
        tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
        tmp84 = tl.where(tmp81, tmp82, tmp83)
        tmp85 = tl.full(tmp84.shape, 0.0, tmp84.dtype)
        tmp86 = tl.where(tmp78, tmp84, tmp85)
        tmp87 = tmp76 + tmp86
        tmp88 = tmp64 + tmp87
        tmp89 = tmp88 - tmp28
        tmp90 = tl.exp(tmp89)
        tmp91 = tmp90 / tmp60
        tl.store(out_ptr2 + (r2 + (256*x3)), tmp91, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/df/cdfcoiehhofl3m6h3fjbjvkt5ynuz7s7saycxdvzxnkuiedbohzg.py
# Source Nodes: [reshape_2], Original ATen: [aten.clone]
# reshape_2 => clone_2
triton_poi_fused_clone_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 65536
    x1 = (xindex // 65536)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (32768 + x0 + (98304*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2k/c2ktt57km4grep72pa3liw6ukabei6ow7acz55pkk3prbckkinkb.py
# Source Nodes: [x_163], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_163 => add_52, mul_92, mul_93, sub_23
triton_poi_fused__native_batch_norm_legit_no_training_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 256
    x2 = (xindex // 65536)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (16384*((x1 + (256*x0)) // 16384)) + (65536*x2) + (x0 % 64)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xp/cxp2srxojnl5ywv6kqbiryth53ag7cc4vter45zqhfpvscqok6oz.py
# Source Nodes: [x_166, x_167], Original ATen: [aten.convolution, aten.silu]
# x_166 => mul_94, sigmoid_24
# x_167 => convolution_28
triton_poi_fused_convolution_silu_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (65536*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/he/chehdhs27c37it2tv5zjp3trhksfxmcommdizjzqgaxz5hk55ir5.py
# Source Nodes: [shortcut_6, x_168, x_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# shortcut_6 => mul_98, sigmoid_25
# x_168 => add_54, mul_96, mul_97, sub_24
# x_174 => add_55
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1024
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
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tl.store(in_out_ptr0 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/px/cpxezupna7wukxrbz5dy6hkuofs3nbdjqrwtzvgf4glwfvoko3yw.py
# Source Nodes: [x_176, x_180, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_176 => add_57, mul_100, mul_101, sub_25
# x_180 => mul_102, sigmoid_26
# x_182 => convolution_30
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_29', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 512
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l4/cl4myjpty46l3agundd6jj7eoerlrahlma6ikxtpfhnz35wb7ri3.py
# Source Nodes: [reshape_12], Original ATen: [aten.clone]
# reshape_12 => clone_7
triton_poi_fused_clone_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16384
    x1 = (xindex // 16384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (163840*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d7/cd75m3nfyiwzgx5t6fi4gh3jczyuhpnpzgscdhnvetbcejtm36io.py
# Source Nodes: [k_3], Original ATen: [aten.clone]
# k_3 => clone_8
triton_poi_fused_clone_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16384
    x1 = (xindex // 16384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (16384 + x0 + (163840*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rh/crhvegsrt7qpz47n3fc32ga5xxwzpvv4q3ehi6kwdk5cdycj2tu6.py
# Source Nodes: [reshape_14], Original ATen: [aten.clone]
# reshape_14 => clone_9
triton_poi_fused_clone_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 131072
    x1 = (xindex // 131072)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (32768 + x0 + (163840*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/g7/cg7o4bdsinnv522fz4xtsvjz555m3dg43lqc5egx7d32klnxnhvz.py
# Source Nodes: [x_191, x_192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d]
# x_191 => avg_pool2d
# x_192 => add_61, mul_105, mul_106, sub_27
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 8
    x2 = (xindex // 4096) % 8
    x3 = (xindex // 32768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((256*x1) + (4096*x2) + (32768*((x1 + (16*x2) + (128*x0)) // 16384)) + (131072*x3) + (((x1 + (16*x2) + (128*x0)) // 128) % 128)), None)
    tmp1 = tl.load(in_ptr0 + (128 + (256*x1) + (4096*x2) + (32768*((1 + (2*x1) + (32*x2) + (256*x0)) // 32768)) + (131072*x3) + (((1 + (2*x1) + (32*x2) + (256*x0)) // 256) % 128)), None)
    tmp3 = tl.load(in_ptr0 + (2048 + (256*x1) + (4096*x2) + (32768*((8 + x1 + (16*x2) + (128*x0)) // 16384)) + (131072*x3) + (((8 + x1 + (16*x2) + (128*x0)) // 128) % 128)), None)
    tmp5 = tl.load(in_ptr0 + (2176 + (256*x1) + (4096*x2) + (32768*((17 + (2*x1) + (32*x2) + (256*x0)) // 32768)) + (131072*x3) + (((17 + (2*x1) + (32*x2) + (256*x0)) // 256) % 128)), None)
    tmp9 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sqrt(tmp13)
    tmp15 = 1 / tmp14
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp10 * tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 + tmp21
    tl.store(out_ptr0 + (x4), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gw/cgw7mllisa36kw2w3xkijssnadgjgjclj3rnfy2d7pucdwsaa72e.py
# Source Nodes: [x_195, x_196], Original ATen: [aten.convolution, aten.silu]
# x_195 => mul_107, sigmoid_27
# x_196 => convolution_31
triton_poi_fused_convolution_silu_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (32768*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/27/c2774ihch7ey7bodasxtjgz6xxb5p7ps5r5z3fqqjhimsavd6rx6.py
# Source Nodes: [shortcut_7, x_197, x_204, x_208], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# shortcut_7 => mul_114, sigmoid_28
# x_197 => add_63, mul_109, mul_110, sub_28
# x_204 => add_65, mul_112, mul_113, sub_29
# x_208 => add_66
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_35', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 2048
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
    tmp29 = tl.sigmoid(tmp28)
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bz/cbzzxrstkbtixsw5ivwiqv5a6gjo6yxjiv554rv2hllfjxm7pyqo.py
# Source Nodes: [x_210, x_214, x_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
# x_210 => add_68, mul_116, mul_117, sub_30
# x_214 => mul_118, sigmoid_29
# x_216 => convolution_34
triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 512
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2u/c2uiury5vr4ptdfvlnznqamau5rt3qyzcg7dslczpk3ad2htd4oj.py
# Source Nodes: [reshape_24], Original ATen: [aten.clone]
# reshape_24 => clone_14
triton_poi_fused_clone_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = (xindex // 4096)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (40960*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6u/c6ukoftohd6ryxkmlq3gxkwpsgmrs2ecwajvsmx6jlcldjokvwvt.py
# Source Nodes: [k_5], Original ATen: [aten.clone]
# k_5 => clone_15
triton_poi_fused_clone_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = (xindex // 4096)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (4096 + x0 + (40960*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/g7/cg7khjececr72qkp6wlcuznjih2n6gpde4tsmnclztjobgrocy3t.py
# Source Nodes: [x_221], Original ATen: [aten.clone]
# x_221 => clone_18
triton_poi_fused_clone_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 16
    x3 = (xindex // 16)
    y0 = yindex % 8
    y1 = (yindex // 8)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (8*x3) + (64*x2) + (64*((y0 + (8*x3)) // 64)) + (1024*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x5 + (128*y4)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zl/czlhrxkr2sebqkasepnx5vpwheg76gc5fs2fg375rpo7kngg5pg6.py
# Source Nodes: [x_217], Original ATen: [aten.clone]
# x_217 => clone_17
triton_poi_fused_clone_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (1024*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (16*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3l/c3lzxwznaaq4um6fxhp26i52fa634hc3bfocuxnuwotuh6gcf7ic.py
# Source Nodes: [attn_4, attn_5, mul_7], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn_4 => add_70
# attn_5 => amax_2, div_2, exp_2, sub_31, sum_3
# mul_7 => mul_119
triton_per_fused__softmax_add_mul_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (r2 + (64*x3)), rmask, other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp3 = 7 + (15*(x0 // 8)) + (r2 // 8)
    tmp4 = tl.full([1, 1], 128, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = (7 + (15*(x0 // 8)) + (r2 // 8)) % 16
    tmp7 = tl.full([1, 1], 15, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr1 + ((15*((7 + (15*(x0 // 8)) + (r2 // 8)) // 16)) + (120*(x0 % 8)) + (960*x1) + ((7 + (15*(x0 // 8)) + (r2 // 8)) % 16)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp15 = 7 + (15*(x0 % 8)) + (r2 % 8)
    tmp16 = tmp15 < tmp4
    tmp17 = (7 + (15*(x0 % 8)) + (r2 % 8)) % 16
    tmp18 = tmp17 < tmp7
    tmp19 = tmp18 & tmp16
    tmp20 = tl.load(in_ptr2 + ((15*(((7 + (15*(x0 % 8)) + (r2 % 8)) // 16) % 8)) + (120*(x0 // 8)) + (960*x1) + ((7 + (15*(x0 % 8)) + (r2 % 8)) % 16)), rmask & tmp19, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp16, tmp22, tmp23)
    tmp25 = tmp14 + tmp24
    tmp26 = tmp2 + tmp25
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
    tmp29 = tl.where(rmask, tmp27, float("-inf"))
    tmp30 = triton_helpers.max2(tmp29, 1)[:, None]
    tmp31 = tmp26 - tmp30
    tmp32 = tl.exp(tmp31)
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = tl.sum(tmp35, 1)[:, None]
    tmp37 = tmp32 / tmp36
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp37, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ok/cokmlw6wl63vbvd634h7yozhyg3k6d22nbj3aju2jygxu4llf2ir.py
# Source Nodes: [reshape_26], Original ATen: [aten.clone]
# reshape_26 => clone_16
triton_poi_fused_clone_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32768
    x1 = (xindex // 32768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (8192 + x0 + (40960*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4j/c4jko622vytzzrsmxytucm7zabuiwwnwo3em5oenl2edq4nmsxi6.py
# Source Nodes: [x_226], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_226 => add_72, mul_121, mul_122, sub_32
triton_poi_fused__native_batch_norm_legit_no_training_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 64
    x2 = (xindex // 32768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((128*x1) + (8192*((x1 + (64*x0)) // 8192)) + (32768*x2) + (x0 % 128)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xa/cxaybzw3sb62e3w322mmjwvbrld5y7d4swtybbjubioyw65j4iii.py
# Source Nodes: [x_231, x_237, x_238, x_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.silu]
# x_231 => add_74, mul_125, mul_126, sub_33
# x_237 => add_75
# x_238 => mul_127, sigmoid_31
# x_241 => mean_5
triton_per_fused__native_batch_norm_legit_no_training_add_mean_silu_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_silu_44', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (r2 + (64*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (r2 + (64*x3)), rmask, other=0.0)
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
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = 64.0
    tmp24 = tmp22 / tmp23
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp24, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1 = args
    args.clear()
    assert_size_stride(arg0_1, (24, ), (1, ))
    assert_size_stride(arg1_1, (24, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, ), (1, ))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (256, ), (1, ))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (64, ), (1, ))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (256, ), (1, ))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (128, ), (1, ))
    assert_size_stride(arg22_1, (128, ), (1, ))
    assert_size_stride(arg23_1, (128, ), (1, ))
    assert_size_stride(arg24_1, (512, ), (1, ))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, ), (1, ))
    assert_size_stride(arg27_1, (512, ), (1, ))
    assert_size_stride(arg28_1, (128, ), (1, ))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (128, ), (1, ))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (512, ), (1, ))
    assert_size_stride(arg33_1, (512, ), (1, ))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (256, ), (1, ))
    assert_size_stride(arg37_1, (256, ), (1, ))
    assert_size_stride(arg38_1, (1024, ), (1, ))
    assert_size_stride(arg39_1, (1024, ), (1, ))
    assert_size_stride(arg40_1, (1024, ), (1, ))
    assert_size_stride(arg41_1, (1024, ), (1, ))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (31, 16), (16, 1))
    assert_size_stride(arg45_1, (31, 16), (16, 1))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (1024, ), (1, ))
    assert_size_stride(arg49_1, (1024, ), (1, ))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (31, 16), (16, 1))
    assert_size_stride(arg53_1, (31, 16), (16, 1))
    assert_size_stride(arg54_1, (512, ), (1, ))
    assert_size_stride(arg55_1, (512, ), (1, ))
    assert_size_stride(arg56_1, (2048, ), (1, ))
    assert_size_stride(arg57_1, (2048, ), (1, ))
    assert_size_stride(arg58_1, (2048, ), (1, ))
    assert_size_stride(arg59_1, (2048, ), (1, ))
    assert_size_stride(arg60_1, (512, ), (1, ))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (15, 16), (16, 1))
    assert_size_stride(arg63_1, (15, 16), (16, 1))
    assert_size_stride(arg64_1, (512, ), (1, ))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (2048, ), (1, ))
    assert_size_stride(arg67_1, (2048, ), (1, ))
    assert_size_stride(arg68_1, (24, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg69_1, (32, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(arg70_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg71_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg72_1, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg73_1, (1, 1, 3), (3, 3, 1))
    assert_size_stride(arg74_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg75_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg76_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg77_1, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg78_1, (1, 1, 3), (3, 3, 1))
    assert_size_stride(arg79_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg80_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg81_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg82_1, (1, 1, 5), (5, 5, 1))
    assert_size_stride(arg83_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg84_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg85_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg86_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg87_1, (1, 1, 5), (5, 5, 1))
    assert_size_stride(arg88_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg89_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg90_1, (256, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg91_1, (1, 1, 5), (5, 5, 1))
    assert_size_stride(arg92_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg93_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg94_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg95_1, (384, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg96_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg97_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg98_1, (640, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg99_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg100_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg101_1, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg102_1, (640, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg103_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg104_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg105_1, (1000, ), (1, ))
    assert_size_stride(arg106_1, (24, ), (1, ))
    assert_size_stride(arg107_1, (24, ), (1, ))
    assert_size_stride(arg108_1, (32, ), (1, ))
    assert_size_stride(arg109_1, (32, ), (1, ))
    assert_size_stride(arg110_1, (64, ), (1, ))
    assert_size_stride(arg111_1, (64, ), (1, ))
    assert_size_stride(arg112_1, (64, ), (1, ))
    assert_size_stride(arg113_1, (64, ), (1, ))
    assert_size_stride(arg114_1, (64, ), (1, ))
    assert_size_stride(arg115_1, (64, ), (1, ))
    assert_size_stride(arg116_1, (256, ), (1, ))
    assert_size_stride(arg117_1, (256, ), (1, ))
    assert_size_stride(arg118_1, (256, ), (1, ))
    assert_size_stride(arg119_1, (256, ), (1, ))
    assert_size_stride(arg120_1, (64, ), (1, ))
    assert_size_stride(arg121_1, (64, ), (1, ))
    assert_size_stride(arg122_1, (64, ), (1, ))
    assert_size_stride(arg123_1, (64, ), (1, ))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (256, ), (1, ))
    assert_size_stride(arg126_1, (128, ), (1, ))
    assert_size_stride(arg127_1, (128, ), (1, ))
    assert_size_stride(arg128_1, (128, ), (1, ))
    assert_size_stride(arg129_1, (128, ), (1, ))
    assert_size_stride(arg130_1, (512, ), (1, ))
    assert_size_stride(arg131_1, (512, ), (1, ))
    assert_size_stride(arg132_1, (512, ), (1, ))
    assert_size_stride(arg133_1, (512, ), (1, ))
    assert_size_stride(arg134_1, (128, ), (1, ))
    assert_size_stride(arg135_1, (128, ), (1, ))
    assert_size_stride(arg136_1, (128, ), (1, ))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (512, ), (1, ))
    assert_size_stride(arg139_1, (512, ), (1, ))
    assert_size_stride(arg140_1, (256, ), (1, ))
    assert_size_stride(arg141_1, (256, ), (1, ))
    assert_size_stride(arg142_1, (256, ), (1, ))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (1024, ), (1, ))
    assert_size_stride(arg145_1, (1024, ), (1, ))
    assert_size_stride(arg146_1, (1024, ), (1, ))
    assert_size_stride(arg147_1, (1024, ), (1, ))
    assert_size_stride(arg148_1, (256, ), (1, ))
    assert_size_stride(arg149_1, (256, ), (1, ))
    assert_size_stride(arg150_1, (256, ), (1, ))
    assert_size_stride(arg151_1, (256, ), (1, ))
    assert_size_stride(arg152_1, (1024, ), (1, ))
    assert_size_stride(arg153_1, (1024, ), (1, ))
    assert_size_stride(arg154_1, (512, ), (1, ))
    assert_size_stride(arg155_1, (512, ), (1, ))
    assert_size_stride(arg156_1, (512, ), (1, ))
    assert_size_stride(arg157_1, (512, ), (1, ))
    assert_size_stride(arg158_1, (2048, ), (1, ))
    assert_size_stride(arg159_1, (2048, ), (1, ))
    assert_size_stride(arg160_1, (2048, ), (1, ))
    assert_size_stride(arg161_1, (2048, ), (1, ))
    assert_size_stride(arg162_1, (512, ), (1, ))
    assert_size_stride(arg163_1, (512, ), (1, ))
    assert_size_stride(arg164_1, (512, ), (1, ))
    assert_size_stride(arg165_1, (512, ), (1, ))
    assert_size_stride(arg166_1, (2048, ), (1, ))
    assert_size_stride(arg167_1, (2048, ), (1, ))
    assert_size_stride(arg168_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg168_1, arg68_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 24, 128, 128), (393216, 16384, 128, 1))
        del arg168_1
        del arg68_1
        buf1 = buf0; del buf0  # reuse
        buf2 = buf1; del buf1  # reuse
        # Source Nodes: [x_1, x_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_0.run(buf2, arg106_1, arg107_1, arg0_1, arg1_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg0_1
        del arg106_1
        del arg107_1
        del arg1_1
        # Source Nodes: [x_4, x_5], Original ATen: [aten.convolution, aten.silu]
        buf3 = extern_kernels.convolution(buf2, arg69_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (8, 32, 128, 128), (524288, 16384, 128, 1))
        del arg69_1
        del buf2
        buf4 = buf3; del buf3  # reuse
        buf5 = buf4; del buf4  # reuse
        # Source Nodes: [x_10, x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_1.run(buf5, arg108_1, arg109_1, arg2_1, arg3_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg108_1
        del arg109_1
        del arg2_1
        del arg3_1
        # Source Nodes: [x_10, x_9], Original ATen: [aten.convolution, aten.silu]
        buf6 = extern_kernels.convolution(buf5, arg70_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        del arg70_1
        del buf5
        buf7 = buf6; del buf6  # reuse
        # Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_2.run(buf7, arg110_1, arg111_1, arg4_1, arg5_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg110_1
        del arg111_1
        del arg4_1
        del arg5_1
        buf8 = empty((8, 64, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_14], Original ATen: [aten.max_pool2d_with_indices, aten.silu]
        triton_poi_fused_max_pool2d_with_indices_silu_3.run(buf7, buf8, 2097152, grid=grid(2097152), stream=stream0)
        del buf7
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg71_1
        buf10 = buf9; del buf9  # reuse
        buf11 = buf10; del buf10  # reuse
        # Source Nodes: [x_17, x_21, x_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_4.run(buf11, arg112_1, arg113_1, arg6_1, arg7_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg112_1
        del arg113_1
        del arg6_1
        del arg7_1
        # Source Nodes: [x_21, x_22], Original ATen: [aten.convolution, aten.silu]
        buf12 = extern_kernels.convolution(buf11, arg72_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf12, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg72_1
        del buf11
        buf13 = buf12; del buf12  # reuse
        buf14 = empty((8, 64), device='cuda', dtype=torch.float32)
        buf15 = reinterpret_tensor(buf14, (8, 1, 64), (64, 64, 1), 0); del buf14  # reuse
        # Source Nodes: [mean, x_23, x_27, y_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.silu]
        triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_silu_5.run(buf13, buf15, arg114_1, arg115_1, arg8_1, arg9_1, 512, 4096, grid=grid(512), stream=stream0)
        del arg114_1
        del arg115_1
        del arg8_1
        del arg9_1
        # Source Nodes: [y_1], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg73_1, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf16, (8, 1, 64), (64, 64, 1))
        del arg73_1
        del buf15
        buf17 = buf13; del buf13  # reuse
        # Source Nodes: [x_27, x_29, x_30], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_mul_silu_6.run(buf17, buf16, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [x_27, x_29, x_30], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf18 = extern_kernels.convolution(buf17, arg74_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg74_1
        del buf17
        # Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf8, arg75_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg75_1
        del buf8
        buf20 = buf18; del buf18  # reuse
        buf21 = buf20; del buf20  # reuse
        # Source Nodes: [shortcut_1, x_31, x_39, x_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_7.run(buf21, arg116_1, arg117_1, arg10_1, arg11_1, buf19, arg118_1, arg119_1, arg12_1, arg13_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg10_1
        del arg116_1
        del arg117_1
        del arg118_1
        del arg119_1
        del arg11_1
        del arg12_1
        del arg13_1
        del buf19
        # Source Nodes: [x_44], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, arg76_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg76_1
        buf23 = buf22; del buf22  # reuse
        buf24 = buf23; del buf23  # reuse
        # Source Nodes: [x_45, x_49, x_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_4.run(buf24, arg120_1, arg121_1, arg14_1, arg15_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg120_1
        del arg121_1
        del arg14_1
        del arg15_1
        # Source Nodes: [x_49, x_50], Original ATen: [aten.convolution, aten.silu]
        buf25 = extern_kernels.convolution(buf24, arg77_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf25, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg77_1
        del buf24
        buf26 = buf25; del buf25  # reuse
        buf27 = reinterpret_tensor(buf16, (8, 64), (64, 1), 0); del buf16  # reuse
        buf28 = reinterpret_tensor(buf27, (8, 1, 64), (64, 64, 1), 0); del buf27  # reuse
        # Source Nodes: [mean_1, x_51, x_55, y_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.silu]
        triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_silu_5.run(buf26, buf28, arg122_1, arg123_1, arg16_1, arg17_1, 512, 4096, grid=grid(512), stream=stream0)
        del arg122_1
        del arg123_1
        del arg16_1
        del arg17_1
        # Source Nodes: [y_4], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, arg78_1, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf29, (8, 1, 64), (64, 64, 1))
        del arg78_1
        del buf28
        buf30 = buf26; del buf26  # reuse
        # Source Nodes: [x_55, x_57, x_58], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_mul_silu_6.run(buf30, buf29, 2097152, grid=grid(2097152), stream=stream0)
        del buf29
        # Source Nodes: [x_55, x_57, x_58], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf31 = extern_kernels.convolution(buf30, arg79_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg79_1
        del buf30
        buf32 = buf21; del buf21  # reuse
        buf33 = buf32; del buf32  # reuse
        # Source Nodes: [shortcut_2, x_59, x_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_8.run(buf33, buf31, arg124_1, arg125_1, arg18_1, arg19_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg124_1
        del arg125_1
        del arg18_1
        del arg19_1
        del buf31
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, arg80_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del arg80_1
        buf35 = buf34; del buf34  # reuse
        buf36 = buf35; del buf35  # reuse
        # Source Nodes: [x_68, x_72, x_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_9.run(buf36, arg126_1, arg127_1, arg20_1, arg21_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg126_1
        del arg127_1
        del arg20_1
        del arg21_1
        # Source Nodes: [x_72, x_73], Original ATen: [aten.convolution, aten.silu]
        buf37 = extern_kernels.convolution(buf36, arg81_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf37, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg81_1
        del buf36
        buf38 = buf37; del buf37  # reuse
        buf39 = empty((8, 128), device='cuda', dtype=torch.float32)
        buf40 = reinterpret_tensor(buf39, (8, 1, 128), (128, 128, 1), 0); del buf39  # reuse
        # Source Nodes: [mean_2, x_74, x_78, y_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_silu_10.run(buf38, buf40, arg128_1, arg129_1, arg22_1, arg23_1, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg128_1
        del arg129_1
        del arg22_1
        del arg23_1
        # Source Nodes: [y_7], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg82_1, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf41, (8, 1, 128), (128, 128, 1))
        del arg82_1
        del buf40
        buf42 = buf38; del buf38  # reuse
        # Source Nodes: [x_78, x_80, x_81], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_mul_silu_11.run(buf42, buf41, 1048576, grid=grid(1048576), stream=stream0)
        # Source Nodes: [x_78, x_80, x_81], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf43 = extern_kernels.convolution(buf42, arg83_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg83_1
        del buf42
        # Source Nodes: [x_89], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf33, arg84_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg84_1
        del buf33
        buf45 = buf43; del buf43  # reuse
        buf46 = buf45; del buf45  # reuse
        # Source Nodes: [shortcut_3, x_82, x_90, x_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_12.run(buf46, arg130_1, arg131_1, arg24_1, arg25_1, buf44, arg132_1, arg133_1, arg26_1, arg27_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg130_1
        del arg131_1
        del arg132_1
        del arg133_1
        del arg24_1
        del arg25_1
        del arg26_1
        del arg27_1
        del buf44
        # Source Nodes: [x_95], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, arg85_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg85_1
        buf48 = buf47; del buf47  # reuse
        buf49 = buf48; del buf48  # reuse
        # Source Nodes: [x_100, x_101, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_13.run(buf49, arg134_1, arg135_1, arg28_1, arg29_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg134_1
        del arg135_1
        del arg28_1
        del arg29_1
        # Source Nodes: [x_100, x_101], Original ATen: [aten.convolution, aten.silu]
        buf50 = extern_kernels.convolution(buf49, arg86_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf50, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg86_1
        del buf49
        buf51 = buf50; del buf50  # reuse
        buf52 = reinterpret_tensor(buf41, (8, 128), (128, 1), 0); del buf41  # reuse
        buf53 = reinterpret_tensor(buf52, (8, 1, 128), (128, 128, 1), 0); del buf52  # reuse
        # Source Nodes: [mean_3, x_102, x_106, y_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_silu_10.run(buf51, buf53, arg136_1, arg137_1, arg30_1, arg31_1, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg136_1
        del arg137_1
        del arg30_1
        del arg31_1
        # Source Nodes: [y_10], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg87_1, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf54, (8, 1, 128), (128, 128, 1))
        del arg87_1
        del buf53
        buf55 = buf51; del buf51  # reuse
        # Source Nodes: [x_106, x_108, x_109], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_mul_silu_11.run(buf55, buf54, 1048576, grid=grid(1048576), stream=stream0)
        del buf54
        # Source Nodes: [x_106, x_108, x_109], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf56 = extern_kernels.convolution(buf55, arg88_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg88_1
        buf57 = buf46; del buf46  # reuse
        buf58 = buf57; del buf57  # reuse
        # Source Nodes: [shortcut_4, x_110, x_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14.run(buf58, buf56, arg138_1, arg139_1, arg32_1, arg33_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg138_1
        del arg139_1
        del arg32_1
        del arg33_1
        del buf56
        # Source Nodes: [x_118], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, arg89_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del arg89_1
        buf60 = buf59; del buf59  # reuse
        buf61 = buf60; del buf60  # reuse
        # Source Nodes: [x_119, x_123, x_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_15.run(buf61, arg140_1, arg141_1, arg34_1, arg35_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg140_1
        del arg141_1
        del arg34_1
        del arg35_1
        # Source Nodes: [x_123, x_124], Original ATen: [aten.convolution, aten.silu]
        buf62 = extern_kernels.convolution(buf61, arg90_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf62, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg90_1
        buf63 = buf62; del buf62  # reuse
        buf64 = empty((8, 256), device='cuda', dtype=torch.float32)
        buf65 = reinterpret_tensor(buf64, (8, 1, 256), (256, 256, 1), 0); del buf64  # reuse
        # Source Nodes: [mean_4, x_125, x_129, y_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_silu_16.run(buf63, buf65, arg142_1, arg143_1, arg36_1, arg37_1, 2048, 256, grid=grid(2048), stream=stream0)
        del arg142_1
        del arg143_1
        del arg36_1
        del arg37_1
        # Source Nodes: [y_13], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, arg91_1, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf66, (8, 1, 256), (256, 256, 1))
        del arg91_1
        del buf65
        buf67 = buf63; del buf63  # reuse
        # Source Nodes: [x_129, x_131, x_132], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_mul_silu_17.run(buf67, buf66, 524288, grid=grid(524288), stream=stream0)
        del buf66
        # Source Nodes: [x_129, x_131, x_132], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf68 = extern_kernels.convolution(buf67, arg92_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg92_1
        # Source Nodes: [x_140], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf58, arg93_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg93_1
        del buf58
        buf70 = buf68; del buf68  # reuse
        buf71 = buf70; del buf70  # reuse
        # Source Nodes: [shortcut_5, x_133, x_141, x_145], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf71, arg144_1, arg145_1, arg38_1, arg39_1, buf69, arg146_1, arg147_1, arg40_1, arg41_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg144_1
        del arg145_1
        del arg146_1
        del arg147_1
        del arg38_1
        del arg39_1
        del arg40_1
        del arg41_1
        # Source Nodes: [x_146], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg94_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg94_1
        buf73 = buf72; del buf72  # reuse
        buf74 = buf73; del buf73  # reuse
        # Source Nodes: [x_147, x_151, x_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_19.run(buf74, arg148_1, arg149_1, arg42_1, arg43_1, 524288, grid=grid(524288), stream=stream0)
        del arg148_1
        del arg149_1
        del arg42_1
        del arg43_1
        # Source Nodes: [x_151, x_153], Original ATen: [aten.convolution, aten.silu]
        buf75 = extern_kernels.convolution(buf74, arg95_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 384, 16, 16), (98304, 256, 16, 1))
        del arg95_1
        buf76 = empty((8, 64, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [reshape], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf75, buf76, 131072, grid=grid(131072), stream=stream0)
        buf77 = empty((8, 64, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf75, buf77, 131072, grid=grid(131072), stream=stream0)
        buf78 = reinterpret_tensor(buf69, (32, 256, 256), (65536, 256, 1), 0); del buf69  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf76, (32, 256, 16), (4096, 1, 256), 0), reinterpret_tensor(buf77, (32, 16, 256), (4096, 256, 1), 0), out=buf78)
        buf79 = reinterpret_tensor(buf77, (32, 16, 16, 16), (4096, 256, 16, 1), 0); del buf77  # reuse
        # Source Nodes: [x_158], Original ATen: [aten.clone]
        triton_poi_fused_clone_22.run(buf76, buf79, 512, 256, grid=grid(512, 256), stream=stream0)
        buf80 = empty((8192, 31), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (8192, 16), (16, 1), 0), reinterpret_tensor(arg45_1, (16, 31), (1, 16), 0), out=buf80)
        del arg45_1
        buf81 = buf79; del buf79  # reuse
        # Source Nodes: [x_154], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf76, buf81, 8192, 16, grid=grid(8192, 16), stream=stream0)
        buf82 = empty((8192, 31), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (8192, 16), (16, 1), 0), reinterpret_tensor(arg44_1, (16, 31), (1, 16), 0), out=buf82)
        del arg44_1
        buf85 = reinterpret_tensor(buf61, (32, 256, 256), (65536, 256, 1), 0); del buf61  # reuse
        # Source Nodes: [attn, attn_1, mul_5], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_red_fused__softmax_add_mul_24.run(buf78, buf80, buf82, buf85, 8192, 256, grid=grid(8192), stream=stream0)
        del buf78
        buf86 = buf74; del buf74  # reuse
        # Source Nodes: [reshape_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf75, buf86, 524288, grid=grid(524288), stream=stream0)
        del buf75
        buf87 = reinterpret_tensor(buf67, (32, 256, 64), (16384, 64, 1), 0); del buf67  # reuse
        # Source Nodes: [attn, attn_1, matmul_3, mul_5], Original ATen: [aten._softmax, aten.add, aten.bmm, aten.mul]
        extern_kernels.bmm(buf85, reinterpret_tensor(buf86, (32, 256, 64), (16384, 1, 256), 0), out=buf87)
        buf88 = reinterpret_tensor(buf86, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf86  # reuse
        # Source Nodes: [x_163], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_26.run(buf87, arg150_1, arg151_1, arg46_1, arg47_1, buf88, 524288, grid=grid(524288), stream=stream0)
        del arg150_1
        del arg151_1
        del arg46_1
        del arg47_1
        buf89 = reinterpret_tensor(buf87, (8, 256, 16, 16), (65536, 256, 16, 1), 0); del buf87  # reuse
        # Source Nodes: [x_166, x_167], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_27.run(buf88, buf89, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del buf88
        # Source Nodes: [x_166, x_167], Original ATen: [aten.convolution, aten.silu]
        buf90 = extern_kernels.convolution(buf89, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg96_1
        del buf89
        buf91 = buf71; del buf71  # reuse
        buf92 = buf91; del buf91  # reuse
        # Source Nodes: [shortcut_6, x_168, x_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_28.run(buf92, buf90, arg152_1, arg153_1, arg48_1, arg49_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg152_1
        del arg153_1
        del arg48_1
        del arg49_1
        # Source Nodes: [x_175], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, arg97_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg97_1
        buf94 = buf93; del buf93  # reuse
        buf95 = buf94; del buf94  # reuse
        # Source Nodes: [x_176, x_180, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_29.run(buf95, arg154_1, arg155_1, arg50_1, arg51_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg154_1
        del arg155_1
        del arg50_1
        del arg51_1
        # Source Nodes: [x_180, x_182], Original ATen: [aten.convolution, aten.silu]
        buf96 = extern_kernels.convolution(buf95, arg98_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 640, 16, 16), (163840, 256, 16, 1))
        del arg98_1
        buf97 = reinterpret_tensor(buf81, (8, 64, 16, 16), (16384, 256, 16, 1), 0); del buf81  # reuse
        # Source Nodes: [reshape_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf96, buf97, 131072, grid=grid(131072), stream=stream0)
        buf98 = buf76; del buf76  # reuse
        # Source Nodes: [k_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf96, buf98, 131072, grid=grid(131072), stream=stream0)
        buf99 = reinterpret_tensor(buf90, (32, 256, 256), (65536, 256, 1), 0); del buf90  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf97, (32, 256, 16), (4096, 1, 256), 0), reinterpret_tensor(buf98, (32, 16, 256), (4096, 256, 1), 0), out=buf99)
        buf100 = reinterpret_tensor(buf98, (32, 16, 16, 16), (4096, 256, 16, 1), 0); del buf98  # reuse
        # Source Nodes: [x_187], Original ATen: [aten.clone]
        triton_poi_fused_clone_22.run(buf97, buf100, 512, 256, grid=grid(512, 256), stream=stream0)
        buf101 = buf82; del buf82  # reuse
        # Source Nodes: [x_187], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (8192, 16), (16, 1), 0), reinterpret_tensor(arg53_1, (16, 31), (1, 16), 0), out=buf101)
        del arg53_1
        buf102 = buf100; del buf100  # reuse
        # Source Nodes: [x_183], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf97, buf102, 8192, 16, grid=grid(8192, 16), stream=stream0)
        buf103 = buf80; del buf80  # reuse
        # Source Nodes: [x_183], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (8192, 16), (16, 1), 0), reinterpret_tensor(arg52_1, (16, 31), (1, 16), 0), out=buf103)
        del arg52_1
        buf106 = buf85; del buf85  # reuse
        # Source Nodes: [attn_2, attn_3, mul_6], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_red_fused__softmax_add_mul_24.run(buf99, buf101, buf103, buf106, 8192, 256, grid=grid(8192), stream=stream0)
        del buf101
        del buf103
        del buf99
        buf107 = buf95; del buf95  # reuse
        # Source Nodes: [reshape_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf96, buf107, 1048576, grid=grid(1048576), stream=stream0)
        del buf96
        buf108 = reinterpret_tensor(buf55, (32, 256, 128), (32768, 128, 1), 0); del buf55  # reuse
        # Source Nodes: [attn_2, attn_3, matmul_7, mul_6], Original ATen: [aten._softmax, aten.add, aten.bmm, aten.mul]
        extern_kernels.bmm(buf106, reinterpret_tensor(buf107, (32, 256, 128), (32768, 1, 256), 0), out=buf108)
        del buf106
        del buf107
        buf109 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_191, x_192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d]
        triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_33.run(buf108, arg156_1, arg157_1, arg54_1, arg55_1, buf109, 262144, grid=grid(262144), stream=stream0)
        del arg156_1
        del arg157_1
        del arg54_1
        del arg55_1
        del buf108
        buf110 = empty((8, 512, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_195, x_196], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf109, buf110, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del buf109
        # Source Nodes: [x_195, x_196], Original ATen: [aten.convolution, aten.silu]
        buf111 = extern_kernels.convolution(buf110, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (8, 2048, 8, 8), (131072, 64, 8, 1))
        del arg99_1
        # Source Nodes: [x_203], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf92, arg100_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 2048, 8, 8), (131072, 64, 8, 1))
        del arg100_1
        del buf92
        buf113 = buf111; del buf111  # reuse
        buf114 = buf113; del buf113  # reuse
        # Source Nodes: [shortcut_7, x_197, x_204, x_208], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_35.run(buf114, arg158_1, arg159_1, arg56_1, arg57_1, buf112, arg160_1, arg161_1, arg58_1, arg59_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg158_1
        del arg159_1
        del arg160_1
        del arg161_1
        del arg56_1
        del arg57_1
        del arg58_1
        del arg59_1
        del buf112
        # Source Nodes: [x_209], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 512, 8, 8), (32768, 64, 8, 1))
        del arg101_1
        buf116 = buf115; del buf115  # reuse
        buf117 = buf116; del buf116  # reuse
        # Source Nodes: [x_210, x_214, x_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_36.run(buf117, arg162_1, arg163_1, arg60_1, arg61_1, 262144, grid=grid(262144), stream=stream0)
        del arg162_1
        del arg163_1
        del arg60_1
        del arg61_1
        # Source Nodes: [x_214, x_216], Original ATen: [aten.convolution, aten.silu]
        buf118 = extern_kernels.convolution(buf117, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 640, 8, 8), (40960, 64, 8, 1))
        del arg102_1
        buf119 = empty((8, 64, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [reshape_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf118, buf119, 32768, grid=grid(32768), stream=stream0)
        buf120 = empty((8, 64, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_38.run(buf118, buf120, 32768, grid=grid(32768), stream=stream0)
        buf121 = reinterpret_tensor(buf102, (32, 64, 64), (4096, 64, 1), 0); del buf102  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf119, (32, 64, 16), (1024, 1, 64), 0), reinterpret_tensor(buf120, (32, 16, 64), (1024, 64, 1), 0), out=buf121)
        buf122 = reinterpret_tensor(buf120, (32, 8, 8, 16), (1024, 128, 16, 1), 0); del buf120  # reuse
        # Source Nodes: [x_221], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf119, buf122, 256, 128, grid=grid(256, 128), stream=stream0)
        buf123 = empty((2048, 15), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_221], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (2048, 16), (16, 1), 0), reinterpret_tensor(arg63_1, (16, 15), (1, 16), 0), out=buf123)
        del arg63_1
        buf124 = buf122; del buf122  # reuse
        # Source Nodes: [x_217], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf119, buf124, 2048, 16, grid=grid(2048, 16), stream=stream0)
        del buf119
        buf125 = empty((2048, 15), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_217], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (2048, 16), (16, 1), 0), reinterpret_tensor(arg62_1, (16, 15), (1, 16), 0), out=buf125)
        del arg62_1
        del buf124
        buf128 = reinterpret_tensor(buf97, (32, 64, 64), (4096, 64, 1), 0); del buf97  # reuse
        # Source Nodes: [attn_4, attn_5, mul_7], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_41.run(buf121, buf123, buf125, buf128, 2048, 64, grid=grid(2048), stream=stream0)
        del buf121
        del buf123
        del buf125
        buf129 = buf117; del buf117  # reuse
        # Source Nodes: [reshape_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_42.run(buf118, buf129, 262144, grid=grid(262144), stream=stream0)
        del buf118
        buf130 = reinterpret_tensor(buf110, (32, 64, 128), (8192, 128, 1), 0); del buf110  # reuse
        # Source Nodes: [attn_4, attn_5, matmul_11, mul_7], Original ATen: [aten._softmax, aten.add, aten.bmm, aten.mul]
        extern_kernels.bmm(buf128, reinterpret_tensor(buf129, (32, 64, 128), (8192, 1, 64), 0), out=buf130)
        del buf128
        buf131 = reinterpret_tensor(buf129, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf129  # reuse
        # Source Nodes: [x_226], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf130, arg164_1, arg165_1, arg64_1, arg65_1, buf131, 262144, grid=grid(262144), stream=stream0)
        del arg164_1
        del arg165_1
        del arg64_1
        del arg65_1
        buf132 = reinterpret_tensor(buf130, (8, 512, 8, 8), (32768, 64, 8, 1), 0); del buf130  # reuse
        # Source Nodes: [x_229, x_230], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf131, buf132, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del buf131
        # Source Nodes: [x_229, x_230], Original ATen: [aten.convolution, aten.silu]
        buf133 = extern_kernels.convolution(buf132, arg103_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (8, 2048, 8, 8), (131072, 64, 8, 1))
        del arg103_1
        del buf132
        buf134 = buf114; del buf114  # reuse
        buf135 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cuda', dtype=torch.float32)
        buf136 = reinterpret_tensor(buf135, (8, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf135  # reuse
        # Source Nodes: [x_231, x_237, x_238, x_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_silu_44.run(buf134, buf136, buf133, arg166_1, arg167_1, arg66_1, arg67_1, 16384, 64, grid=grid(16384), stream=stream0)
        del arg166_1
        del arg167_1
        del arg66_1
        del arg67_1
        del buf133
        del buf134
        buf137 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_245], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg105_1, reinterpret_tensor(buf136, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg104_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf137)
        del arg104_1
        del arg105_1
        return (buf137, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((31, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((31, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((31, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((31, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((15, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((15, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((32, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1, 1, 3), (3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1, 1, 3), (3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1, 1, 5), (5, 5, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1, 1, 5), (5, 5, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1, 1, 5), (5, 5, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((384, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((640, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((640, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('eca_botnext26ts_256', benchmark_compiled_module)
