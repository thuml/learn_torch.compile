
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


# kernel path: /tmp/torchinductor_youkaichao/7o/c7on5wcadzsolmecdmtk54wxnwcjivabulofkhsllh63is2bioi7.py
# Source Nodes: [x_147, x_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_147 => add_48, mul_87, mul_88, sub_21
# x_151 => mul_89, sigmoid_23
triton_poi_fused__native_batch_norm_legit_no_training_silu_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_19', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/u5/cu5jrruqu53lttja2orsuzqjspvmd7p6isingggtosn3eauff5xl.py
# Source Nodes: [matmul, q_3], Original ATen: [aten.clone]
# matmul => clone_2
# q_3 => clone_4
triton_poi_fused_clone_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64) % 4
    y2 = (yindex // 256)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + ((8*(y1 % 2)) + (16*(y0 // 8)) + (128*(y1 // 2)) + (256*x3) + (4096*y2) + (y0 % 8)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + (16*y4)), tmp0, xmask)
    tl.store(out_ptr1 + (x3 + (16*y4)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nt/cnt3c6x6vvpndxwp34ikorto3qfuk4gotb5ke6fp2qpe3jwmbn3i.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_3
triton_poi_fused_clone_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 144
    x2 = (xindex // 2304) % 4
    x1 = (xindex // 144) % 16
    x3 = (xindex // 9216)
    x4 = xindex
    tmp0 = (-2) + (8*((x0 + (144*x2)) // 288)) + (x0 // 12)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + (8*(x2 % 2)) + (x0 % 12)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-34) + (8*(x2 % 2)) + (16*(x0 // 12)) + (128*((x0 + (144*x2)) // 288)) + (256*x1) + (256*((x0 + (144*x2)) // 576)) + (12288*x3) + (x0 % 12)), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zh/czh6uytpfjtfi6ud2phtcz4llshyltkibltanrtzmbvghrbrfh4y.py
# Source Nodes: [x_157], Original ATen: [aten.clone]
# x_157 => clone_5
triton_poi_fused_clone_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (y0 + (8*((y1 % 4) % 2)) + (16*x3) + (128*((y1 % 4) // 2)) + (256*x2) + (4096*(y1 // 4))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x5 + (128*y4)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kl/cklisdqi2aofp3xihyikpepynk2i3g4evywahrdpqflzd57yp4dw.py
# Source Nodes: [attn, attn_1, mul_5], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn => add_50
# attn_1 => amax, div, exp, sub_22, sum_1
# mul_5 => mul_90
triton_red_fused__softmax_add_mul_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_mul_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp28 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (144*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.25
        tmp2 = tmp0 * tmp1
        tmp3 = 11 + (23*(x0 // 8)) + (r2 // 12)
        tmp4 = tl.full([1, 1], 192, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = (11 + (23*(x0 // 8)) + (r2 // 12)) % 24
        tmp7 = tl.full([1, 1], 23, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp5
        tmp10 = tl.load(in_ptr1 + ((23*((11 + (23*(x0 // 8)) + (r2 // 12)) // 24)) + (184*(x0 % 8)) + (1472*x1) + ((11 + (23*(x0 // 8)) + (r2 // 12)) % 24)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp9, tmp10, tmp11)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp5, tmp12, tmp13)
        tmp15 = 11 + (23*(x0 % 8)) + (r2 % 12)
        tmp16 = tmp15 < tmp4
        tmp17 = (11 + (23*(x0 % 8)) + (r2 % 12)) % 24
        tmp18 = tmp17 < tmp7
        tmp19 = tmp18 & tmp16
        tmp20 = tl.load(in_ptr2 + ((23*(((11 + (23*(x0 % 8)) + (r2 % 12)) // 24) % 8)) + (184*(x0 // 8)) + (1472*x1) + ((11 + (23*(x0 % 8)) + (r2 % 12)) % 24)), rmask & tmp19, eviction_policy='evict_last', other=0.0)
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
        tmp30 = tl.load(in_ptr0 + (r2 + (144*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = 0.25
        tmp32 = tmp30 * tmp31
        tmp33 = 11 + (23*(x0 // 8)) + (r2 // 12)
        tmp34 = tl.full([1, 1], 192, tl.int64)
        tmp35 = tmp33 < tmp34
        tmp36 = (11 + (23*(x0 // 8)) + (r2 // 12)) % 24
        tmp37 = tl.full([1, 1], 23, tl.int64)
        tmp38 = tmp36 < tmp37
        tmp39 = tmp38 & tmp35
        tmp40 = tl.load(in_ptr1 + ((23*((11 + (23*(x0 // 8)) + (r2 // 12)) // 24)) + (184*(x0 % 8)) + (1472*x1) + ((11 + (23*(x0 // 8)) + (r2 // 12)) % 24)), rmask & tmp39, eviction_policy='evict_last', other=0.0)
        tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
        tmp42 = tl.where(tmp39, tmp40, tmp41)
        tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
        tmp44 = tl.where(tmp35, tmp42, tmp43)
        tmp45 = 11 + (23*(x0 % 8)) + (r2 % 12)
        tmp46 = tmp45 < tmp34
        tmp47 = (11 + (23*(x0 % 8)) + (r2 % 12)) % 24
        tmp48 = tmp47 < tmp37
        tmp49 = tmp48 & tmp46
        tmp50 = tl.load(in_ptr2 + ((23*(((11 + (23*(x0 % 8)) + (r2 % 12)) // 24) % 8)) + (184*(x0 // 8)) + (1472*x1) + ((11 + (23*(x0 % 8)) + (r2 % 12)) % 24)), rmask & tmp49, eviction_policy='evict_last', other=0.0)
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
        tmp62 = tl.load(in_ptr0 + (r2 + (144*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp63 = 0.25
        tmp64 = tmp62 * tmp63
        tmp65 = 11 + (23*(x0 // 8)) + (r2 // 12)
        tmp66 = tl.full([1, 1], 192, tl.int64)
        tmp67 = tmp65 < tmp66
        tmp68 = (11 + (23*(x0 // 8)) + (r2 // 12)) % 24
        tmp69 = tl.full([1, 1], 23, tl.int64)
        tmp70 = tmp68 < tmp69
        tmp71 = tmp70 & tmp67
        tmp72 = tl.load(in_ptr1 + ((23*((11 + (23*(x0 // 8)) + (r2 // 12)) // 24)) + (184*(x0 % 8)) + (1472*x1) + ((11 + (23*(x0 // 8)) + (r2 // 12)) % 24)), rmask & tmp71, eviction_policy='evict_last', other=0.0)
        tmp73 = tl.full(tmp72.shape, 0.0, tmp72.dtype)
        tmp74 = tl.where(tmp71, tmp72, tmp73)
        tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
        tmp76 = tl.where(tmp67, tmp74, tmp75)
        tmp77 = 11 + (23*(x0 % 8)) + (r2 % 12)
        tmp78 = tmp77 < tmp66
        tmp79 = (11 + (23*(x0 % 8)) + (r2 % 12)) % 24
        tmp80 = tmp79 < tmp69
        tmp81 = tmp80 & tmp78
        tmp82 = tl.load(in_ptr2 + ((23*(((11 + (23*(x0 % 8)) + (r2 % 12)) // 24) % 8)) + (184*(x0 // 8)) + (1472*x1) + ((11 + (23*(x0 % 8)) + (r2 % 12)) % 24)), rmask & tmp81, eviction_policy='evict_last', other=0.0)
        tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
        tmp84 = tl.where(tmp81, tmp82, tmp83)
        tmp85 = tl.full(tmp84.shape, 0.0, tmp84.dtype)
        tmp86 = tl.where(tmp78, tmp84, tmp85)
        tmp87 = tmp76 + tmp86
        tmp88 = tmp64 + tmp87
        tmp89 = tmp88 - tmp28
        tmp90 = tl.exp(tmp89)
        tmp91 = tmp90 / tmp60
        tl.store(out_ptr2 + (r2 + (144*x3)), tmp91, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dr/cdrxtqmfqyd67bsj6b6pgxg2ohb6uc23azk2rvdkb4fluaatpote.py
# Source Nodes: [matmul_3], Original ATen: [aten.clone]
# matmul_3 => clone_7
triton_poi_fused_clone_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 32], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 36864
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 144
    y1 = (yindex // 144) % 4
    x3 = xindex
    y2 = (yindex // 576)
    y4 = yindex
    tmp0 = (-2) + (8*((y0 + (144*y1)) // 288)) + (y0 // 12)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + (8*(y1 % 2)) + (y0 % 12)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (4062 + (8*(y1 % 2)) + (16*(y0 // 12)) + (128*((y0 + (144*y1)) // 288)) + (256*x3) + (256*((y0 + (144*y1)) // 576)) + (12288*y2) + (y0 % 12)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tl.store(out_ptr0 + (x3 + (32*y4)), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f3/cf3u7gr5lt2lpuyy4tsrupxohc4n6ehtxg2g2r3rl4vg4h3ijoej.py
# Source Nodes: [x_162], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_162 => add_52, mul_92, mul_93, sub_23
triton_poi_fused__native_batch_norm_legit_no_training_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 16
    x2 = (xindex // 4096) % 16
    x3 = (xindex // 65536)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*(x1 % 8)) + (256*(x2 % 8)) + (2048*(x1 // 8)) + (4096*(x2 // 8)) + (8192*(x0 // 32)) + (65536*x3) + (x0 % 32)), None)
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
    tl.store(out_ptr0 + (x4), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pe/cpeatp3qyhzuadnp42embslqe6ap3t7lorzn6h3bf46feb6khrh4.py
# Source Nodes: [x_165, x_166], Original ATen: [aten.convolution, aten.silu]
# x_165 => mul_94, sigmoid_24
# x_166 => convolution_29
triton_poi_fused_convolution_silu_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_26', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/y4/cy4qrtzgfr7npfvcwtui6q6wsboniez4wg7fdwp5x4aqt5ehs4ga.py
# Source Nodes: [shortcut_6, x_167, x_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# shortcut_6 => mul_98, sigmoid_25
# x_167 => add_54, mul_96, mul_97, sub_24
# x_173 => add_55
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_27', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/ii/ciibikyrbgbzp6d75hq7osdsq7uf3gpy62a7how44gzkms24gsiv.py
# Source Nodes: [x_175, x_179], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_175 => add_57, mul_100, mul_101, sub_25
# x_179 => mul_102, sigmoid_26
triton_poi_fused__native_batch_norm_legit_no_training_silu_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_28', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/4p/c4ph2oec2xslrwfifqf2w4abqbfjvoxlnqxhzjk734aqstj3fzyw.py
# Source Nodes: [matmul_4, q_8], Original ATen: [aten.clone]
# matmul_4 => clone_12
# q_8 => clone_14
triton_poi_fused_clone_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 4
    y2 = (yindex // 64)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + ((4*(y1 % 2)) + (8*(y0 // 4)) + (32*(y1 // 2)) + (64*x3) + (1024*y2) + (y0 % 4)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + (16*y4)), tmp0, xmask)
    tl.store(out_ptr1 + (x3 + (16*y4)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5l/c5litu64eubknvqnsbs2eu5kpbdbaoqvlnynqwmk2i4yjjspzq22.py
# Source Nodes: [matmul_4], Original ATen: [aten.clone]
# matmul_4 => clone_13
triton_poi_fused_clone_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 144
    x2 = (xindex // 2304) % 4
    x1 = (xindex // 144) % 16
    x3 = (xindex // 9216)
    x4 = xindex
    tmp0 = (-2) + (8*((x0 + (144*x2)) // 288)) + (x0 // 12)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + (8*(x2 % 2)) + (x0 % 12)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-34) + (8*(x2 % 2)) + (16*(x0 // 12)) + (128*((x0 + (144*x2)) // 288)) + (256*x1) + (256*((x0 + (144*x2)) // 576)) + (20480*x3) + (x0 % 12)), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/55/c55cw3rh2g4a7fddkumhgygdhgrwqflyynmf26oe7wdxh6eszzje.py
# Source Nodes: [x_185], Original ATen: [aten.clone]
# x_185 => clone_15
triton_poi_fused_clone_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 16
    x3 = (xindex // 16)
    y0 = yindex % 4
    y1 = (yindex // 4)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (4*((y1 % 4) % 2)) + (8*x3) + (32*((y1 % 4) // 2)) + (64*x2) + (1024*(y1 // 4))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x5 + (64*y4)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dh/cdhflsbrz2yamkd5ogokipbzujtwp24i6conflxc6yjx42b37ocq.py
# Source Nodes: [attn_2, attn_3, mul_6], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn_2 => add_59
# attn_3 => amax_1, div_1, exp_1, sub_26, sum_2
# mul_6 => mul_103
triton_per_fused__softmax_add_mul_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp0 = tl.load(in_ptr0 + (r2 + (144*x3)), rmask, other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp3 = 11 + (23*(x0 // 4)) + (r2 // 12)
    tmp4 = tl.full([1, 1], 96, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = (11 + (23*(x0 // 4)) + (r2 // 12)) % 24
    tmp7 = tl.full([1, 1], 23, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr1 + ((23*((11 + (23*(x0 // 4)) + (r2 // 12)) // 24)) + (92*(x0 % 4)) + (368*x1) + ((11 + (23*(x0 // 4)) + (r2 // 12)) % 24)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp15 = 11 + (23*(x0 % 4)) + (r2 % 12)
    tmp16 = tmp15 < tmp4
    tmp17 = (11 + (23*(x0 % 4)) + (r2 % 12)) % 24
    tmp18 = tmp17 < tmp7
    tmp19 = tmp18 & tmp16
    tmp20 = tl.load(in_ptr2 + ((23*(((11 + (23*(x0 % 4)) + (r2 % 12)) // 24) % 4)) + (92*(x0 // 4)) + (368*x1) + ((11 + (23*(x0 % 4)) + (r2 % 12)) % 24)), rmask & tmp19, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr2 + (r2 + (144*x3)), tmp37, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hk/chkmt752icdf6uufrwhh2bx3gx5k2b2iq3n6nakxbfyni642wppw.py
# Source Nodes: [matmul_7], Original ATen: [aten.clone]
# matmul_7 => clone_17
triton_poi_fused_clone_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 36864
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 144
    y1 = (yindex // 144) % 4
    x3 = xindex
    y2 = (yindex // 576)
    y4 = yindex
    tmp0 = (-2) + (8*((y0 + (144*y1)) // 288)) + (y0 // 12)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + (8*(y1 % 2)) + (y0 % 12)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (4062 + (8*(y1 % 2)) + (16*(y0 // 12)) + (128*((y0 + (144*y1)) // 288)) + (256*x3) + (256*((y0 + (144*y1)) // 576)) + (20480*y2) + (y0 % 12)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tl.store(out_ptr0 + (x3 + (64*y4)), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gq/cgq4b4uoppmo6iwyl36zdgwxzeptbegmmjmtcf2lxnxcthg2ybsc.py
# Source Nodes: [x_190], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_190 => add_61, mul_105, mul_106, sub_27
triton_poi_fused__native_batch_norm_legit_no_training_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_34', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 4)) + (256*(x2 % 4)) + (1024*(x1 // 4)) + (2048*(x2 // 4)) + (4096*(x0 // 64)) + (32768*x3) + (x0 % 64)), None)
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
    tl.store(out_ptr0 + (x4), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2e/c2e3afpoheyoilahybhxxjnfgirsrkeyvommio3mpvsckkusiuqy.py
# Source Nodes: [x_193, x_194], Original ATen: [aten.convolution, aten.silu]
# x_193 => mul_107, sigmoid_27
# x_194 => convolution_33
triton_poi_fused_convolution_silu_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_35', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxohkv46p5c73f5fmfbfyg7biq5tiwfq77tbxeauwgehp4cvwpd.py
# Source Nodes: [shortcut_7, x_195, x_202, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
# shortcut_7 => mul_114, sigmoid_28
# x_195 => add_63, mul_109, mul_110, sub_28
# x_202 => add_65, mul_112, mul_113, sub_29
# x_206 => add_66
triton_poi_fused__native_batch_norm_legit_no_training_add_silu_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_silu_36', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/6y/c6yiaqwq22pglikultyptduccamgjupbeom67moktwwun5wgo37d.py
# Source Nodes: [x_208, x_212], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_208 => add_68, mul_116, mul_117, sub_30
# x_212 => mul_118, sigmoid_29
triton_poi_fused__native_batch_norm_legit_no_training_silu_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_37', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/b4/cb4zte37swjuonaa4yb7k42oze2puzmm4hoqb23nmmcxbj636wex.py
# Source Nodes: [kv_7], Original ATen: [aten.constant_pad_nd]
# kv_7 => constant_pad_nd_10
triton_poi_fused_constant_pad_nd_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 737280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 12) % 12
    x0 = xindex % 12
    x2 = (xindex // 144)
    x4 = xindex
    tmp0 = (-2) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-18) + x0 + (8*x1) + (64*x2)), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tt/cttxgcjes5tg6udzicttrp745tis32nfg6mpyqfsk7cbb6q35sme.py
# Source Nodes: [matmul_8], Original ATen: [aten.bmm]
# matmul_8 => bmm_4
triton_poi_fused_bmm_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2304
    x1 = (xindex // 2304)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (11520*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ng/cngxcvmorfuje524lgemjm2d4kbcb6qtjioqlcvz2uligde7su4y.py
# Source Nodes: [x_218], Original ATen: [aten.clone]
# x_218 => clone_21
triton_poi_fused_clone_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
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
    tmp0 = tl.load(in_ptr0 + (y0 + (8*x3) + (64*x2) + (1024*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x5 + (128*y4)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m2/cm2fxjktd7z2wv7bqd7uequi7upvrhvrfok4xrrb53ysuvzf7zgc.py
# Source Nodes: [x_214], Original ATen: [aten.clone]
# x_214 => clone_20
triton_poi_fused_clone_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
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


# kernel path: /tmp/torchinductor_youkaichao/ap/capaqaqkgesfmwry7dw3irucv2vhcpxk4bsqldv4ul4lilitltid.py
# Source Nodes: [attn_4, attn_5, mul_7], Original ATen: [aten._softmax, aten.add, aten.mul]
# attn_4 => add_70
# attn_5 => amax_2, div_2, exp_2, sub_31, sum_3
# mul_7 => mul_119
triton_per_fused__softmax_add_mul_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (r2 + (144*x3)), rmask, other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp3 = 11 + (23*(x0 // 8)) + (r2 // 12)
    tmp4 = tl.full([1, 1], 192, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = (11 + (23*(x0 // 8)) + (r2 // 12)) % 24
    tmp7 = tl.full([1, 1], 23, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr1 + ((23*((11 + (23*(x0 // 8)) + (r2 // 12)) // 24)) + (184*(x0 % 8)) + (1472*x1) + ((11 + (23*(x0 // 8)) + (r2 // 12)) % 24)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp15 = 11 + (23*(x0 % 8)) + (r2 % 12)
    tmp16 = tmp15 < tmp4
    tmp17 = (11 + (23*(x0 % 8)) + (r2 % 12)) % 24
    tmp18 = tmp17 < tmp7
    tmp19 = tmp18 & tmp16
    tmp20 = tl.load(in_ptr2 + ((23*(((11 + (23*(x0 % 8)) + (r2 % 12)) // 24) % 8)) + (184*(x0 // 8)) + (1472*x1) + ((11 + (23*(x0 % 8)) + (r2 % 12)) % 24)), rmask & tmp19, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr2 + (r2 + (144*x3)), tmp37, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pr/cprkbgwbr57p2kcsxdh6ouwovjonjazs32vpwekh24lh5cshn2q2.py
# Source Nodes: [matmul_11], Original ATen: [aten.bmm]
# matmul_11 => bmm_5
triton_poi_fused_bmm_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 9216
    x1 = (xindex // 9216)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2304 + x0 + (11520*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ar/carq4o4weyqwz2ewnilh27xbgktsos2n4pukymldpzf7kzrafly2.py
# Source Nodes: [x_223], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_223 => add_72, mul_121, mul_122, sub_32
triton_poi_fused__native_batch_norm_legit_no_training_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_44', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (4096*(x0 // 64)) + (32768*x2) + (x0 % 64)), None)
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


# kernel path: /tmp/torchinductor_youkaichao/j7/cj7hi7vki2ewwrqmfvq5nifon4iyyk7exzxj6o4ggvsoabbkulsi.py
# Source Nodes: [x_228, x_234, x_235, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.silu]
# x_228 => add_74, mul_125, mul_126, sub_33
# x_234 => add_75
# x_235 => mul_127, sigmoid_31
# x_238 => mean_5
triton_per_fused__native_batch_norm_legit_no_training_add_mean_silu_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_silu_45', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1 = args
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
    assert_size_stride(arg44_1, (23, 16), (16, 1))
    assert_size_stride(arg45_1, (23, 16), (16, 1))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (1024, ), (1, ))
    assert_size_stride(arg49_1, (1024, ), (1, ))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (23, 16), (16, 1))
    assert_size_stride(arg53_1, (23, 16), (16, 1))
    assert_size_stride(arg54_1, (512, ), (1, ))
    assert_size_stride(arg55_1, (512, ), (1, ))
    assert_size_stride(arg56_1, (2048, ), (1, ))
    assert_size_stride(arg57_1, (2048, ), (1, ))
    assert_size_stride(arg58_1, (2048, ), (1, ))
    assert_size_stride(arg59_1, (2048, ), (1, ))
    assert_size_stride(arg60_1, (512, ), (1, ))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (23, 16), (16, 1))
    assert_size_stride(arg63_1, (23, 16), (16, 1))
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
    assert_size_stride(arg95_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg96_1, (384, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg97_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg98_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg99_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg100_1, (640, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg101_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg102_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg103_1, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg104_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg105_1, (640, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg106_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg107_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg108_1, (1000, ), (1, ))
    assert_size_stride(arg109_1, (24, ), (1, ))
    assert_size_stride(arg110_1, (24, ), (1, ))
    assert_size_stride(arg111_1, (32, ), (1, ))
    assert_size_stride(arg112_1, (32, ), (1, ))
    assert_size_stride(arg113_1, (64, ), (1, ))
    assert_size_stride(arg114_1, (64, ), (1, ))
    assert_size_stride(arg115_1, (64, ), (1, ))
    assert_size_stride(arg116_1, (64, ), (1, ))
    assert_size_stride(arg117_1, (64, ), (1, ))
    assert_size_stride(arg118_1, (64, ), (1, ))
    assert_size_stride(arg119_1, (256, ), (1, ))
    assert_size_stride(arg120_1, (256, ), (1, ))
    assert_size_stride(arg121_1, (256, ), (1, ))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (64, ), (1, ))
    assert_size_stride(arg124_1, (64, ), (1, ))
    assert_size_stride(arg125_1, (64, ), (1, ))
    assert_size_stride(arg126_1, (64, ), (1, ))
    assert_size_stride(arg127_1, (256, ), (1, ))
    assert_size_stride(arg128_1, (256, ), (1, ))
    assert_size_stride(arg129_1, (128, ), (1, ))
    assert_size_stride(arg130_1, (128, ), (1, ))
    assert_size_stride(arg131_1, (128, ), (1, ))
    assert_size_stride(arg132_1, (128, ), (1, ))
    assert_size_stride(arg133_1, (512, ), (1, ))
    assert_size_stride(arg134_1, (512, ), (1, ))
    assert_size_stride(arg135_1, (512, ), (1, ))
    assert_size_stride(arg136_1, (512, ), (1, ))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (128, ), (1, ))
    assert_size_stride(arg139_1, (128, ), (1, ))
    assert_size_stride(arg140_1, (128, ), (1, ))
    assert_size_stride(arg141_1, (512, ), (1, ))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (256, ), (1, ))
    assert_size_stride(arg147_1, (1024, ), (1, ))
    assert_size_stride(arg148_1, (1024, ), (1, ))
    assert_size_stride(arg149_1, (1024, ), (1, ))
    assert_size_stride(arg150_1, (1024, ), (1, ))
    assert_size_stride(arg151_1, (256, ), (1, ))
    assert_size_stride(arg152_1, (256, ), (1, ))
    assert_size_stride(arg153_1, (256, ), (1, ))
    assert_size_stride(arg154_1, (256, ), (1, ))
    assert_size_stride(arg155_1, (1024, ), (1, ))
    assert_size_stride(arg156_1, (1024, ), (1, ))
    assert_size_stride(arg157_1, (512, ), (1, ))
    assert_size_stride(arg158_1, (512, ), (1, ))
    assert_size_stride(arg159_1, (512, ), (1, ))
    assert_size_stride(arg160_1, (512, ), (1, ))
    assert_size_stride(arg161_1, (2048, ), (1, ))
    assert_size_stride(arg162_1, (2048, ), (1, ))
    assert_size_stride(arg163_1, (2048, ), (1, ))
    assert_size_stride(arg164_1, (2048, ), (1, ))
    assert_size_stride(arg165_1, (512, ), (1, ))
    assert_size_stride(arg166_1, (512, ), (1, ))
    assert_size_stride(arg167_1, (512, ), (1, ))
    assert_size_stride(arg168_1, (512, ), (1, ))
    assert_size_stride(arg169_1, (2048, ), (1, ))
    assert_size_stride(arg170_1, (2048, ), (1, ))
    assert_size_stride(arg171_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg171_1, arg68_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 24, 128, 128), (393216, 16384, 128, 1))
        del arg171_1
        del arg68_1
        buf1 = buf0; del buf0  # reuse
        buf2 = buf1; del buf1  # reuse
        # Source Nodes: [x_1, x_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_0.run(buf2, arg109_1, arg110_1, arg0_1, arg1_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg0_1
        del arg109_1
        del arg110_1
        del arg1_1
        # Source Nodes: [x_4, x_5], Original ATen: [aten.convolution, aten.silu]
        buf3 = extern_kernels.convolution(buf2, arg69_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (8, 32, 128, 128), (524288, 16384, 128, 1))
        del arg69_1
        del buf2
        buf4 = buf3; del buf3  # reuse
        buf5 = buf4; del buf4  # reuse
        # Source Nodes: [x_10, x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_1.run(buf5, arg111_1, arg112_1, arg2_1, arg3_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg111_1
        del arg112_1
        del arg2_1
        del arg3_1
        # Source Nodes: [x_10, x_9], Original ATen: [aten.convolution, aten.silu]
        buf6 = extern_kernels.convolution(buf5, arg70_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        del arg70_1
        del buf5
        buf7 = buf6; del buf6  # reuse
        # Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_2.run(buf7, arg113_1, arg114_1, arg4_1, arg5_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg113_1
        del arg114_1
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
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_4.run(buf11, arg115_1, arg116_1, arg6_1, arg7_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg115_1
        del arg116_1
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
        triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_silu_5.run(buf13, buf15, arg117_1, arg118_1, arg8_1, arg9_1, 512, 4096, grid=grid(512), stream=stream0)
        del arg117_1
        del arg118_1
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
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_7.run(buf21, arg119_1, arg120_1, arg10_1, arg11_1, buf19, arg121_1, arg122_1, arg12_1, arg13_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg10_1
        del arg119_1
        del arg11_1
        del arg120_1
        del arg121_1
        del arg122_1
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
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_4.run(buf24, arg123_1, arg124_1, arg14_1, arg15_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg123_1
        del arg124_1
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
        triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_silu_5.run(buf26, buf28, arg125_1, arg126_1, arg16_1, arg17_1, 512, 4096, grid=grid(512), stream=stream0)
        del arg125_1
        del arg126_1
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
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_8.run(buf33, buf31, arg127_1, arg128_1, arg18_1, arg19_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg127_1
        del arg128_1
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
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_9.run(buf36, arg129_1, arg130_1, arg20_1, arg21_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg129_1
        del arg130_1
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
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_silu_10.run(buf38, buf40, arg131_1, arg132_1, arg22_1, arg23_1, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg131_1
        del arg132_1
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
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_12.run(buf46, arg133_1, arg134_1, arg24_1, arg25_1, buf44, arg135_1, arg136_1, arg26_1, arg27_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg133_1
        del arg134_1
        del arg135_1
        del arg136_1
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
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_13.run(buf49, arg137_1, arg138_1, arg28_1, arg29_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg137_1
        del arg138_1
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
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_silu_10.run(buf51, buf53, arg139_1, arg140_1, arg30_1, arg31_1, 1024, 1024, grid=grid(1024), stream=stream0)
        del arg139_1
        del arg140_1
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
        del buf55
        buf57 = buf46; del buf46  # reuse
        buf58 = buf57; del buf57  # reuse
        # Source Nodes: [shortcut_4, x_110, x_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_14.run(buf58, buf56, arg141_1, arg142_1, arg32_1, arg33_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg141_1
        del arg142_1
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
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_silu_15.run(buf61, arg143_1, arg144_1, arg34_1, arg35_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg143_1
        del arg144_1
        del arg34_1
        del arg35_1
        # Source Nodes: [x_123, x_124], Original ATen: [aten.convolution, aten.silu]
        buf62 = extern_kernels.convolution(buf61, arg90_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf62, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg90_1
        del buf61
        buf63 = buf62; del buf62  # reuse
        buf64 = empty((8, 256), device='cuda', dtype=torch.float32)
        buf65 = reinterpret_tensor(buf64, (8, 1, 256), (256, 256, 1), 0); del buf64  # reuse
        # Source Nodes: [mean_4, x_125, x_129, y_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_convolution_mean_silu_16.run(buf63, buf65, arg145_1, arg146_1, arg36_1, arg37_1, 2048, 256, grid=grid(2048), stream=stream0)
        del arg145_1
        del arg146_1
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
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_18.run(buf71, arg147_1, arg148_1, arg38_1, arg39_1, buf69, arg149_1, arg150_1, arg40_1, arg41_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del arg150_1
        del arg38_1
        del arg39_1
        del arg40_1
        del arg41_1
        del buf69
        # Source Nodes: [x_146], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg94_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg94_1
        buf73 = buf72; del buf72  # reuse
        buf74 = buf73; del buf73  # reuse
        # Source Nodes: [x_147, x_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_19.run(buf74, arg151_1, arg152_1, arg42_1, arg43_1, 524288, grid=grid(524288), stream=stream0)
        del arg151_1
        del arg152_1
        del arg42_1
        del arg43_1
        # Source Nodes: [kv], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 384, 16, 16), (98304, 256, 16, 1))
        del arg96_1
        # Source Nodes: [q], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf74, arg95_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 128, 16, 16), (32768, 256, 16, 1))
        del arg95_1
        buf77 = empty((64, 4, 64, 16), device='cuda', dtype=torch.float32)
        buf82 = empty((64, 4, 64, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul, q_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf76, buf77, buf82, 16384, 16, grid=grid(16384, 16), stream=stream0)
        buf78 = empty((64, 4, 16, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf75, buf78, 589824, grid=grid(589824), stream=stream0)
        buf79 = empty((256, 64, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf77, (256, 64, 16), (1024, 16, 1), 0), reinterpret_tensor(buf78, (256, 16, 144), (2304, 144, 1), 0), out=buf79)
        buf80 = reinterpret_tensor(buf77, (256, 8, 8, 16), (1024, 128, 16, 1), 0); del buf77  # reuse
        # Source Nodes: [x_157], Original ATen: [aten.clone]
        triton_poi_fused_clone_22.run(buf76, buf80, 2048, 128, grid=grid(2048, 128), stream=stream0)
        del buf76
        buf81 = empty((16384, 23), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_157], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (16384, 16), (16, 1), 0), reinterpret_tensor(arg45_1, (16, 23), (1, 16), 0), out=buf81)
        del arg45_1
        buf83 = empty((16384, 23), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_153], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (16384, 16), (16, 1), 0), reinterpret_tensor(arg44_1, (16, 23), (1, 16), 0), out=buf83)
        del arg44_1
        buf86 = empty((64, 4, 64, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn, attn_1, mul_5], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_red_fused__softmax_add_mul_23.run(buf79, buf81, buf83, buf86, 16384, 144, grid=grid(16384), stream=stream0)
        del buf79
        del buf81
        del buf83
        buf87 = empty((64, 4, 144, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf75, buf87, 36864, 32, grid=grid(36864, 32), stream=stream0)
        del buf75
        buf88 = reinterpret_tensor(buf74, (256, 64, 32), (2048, 32, 1), 0); del buf74  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf86, (256, 64, 144), (9216, 144, 1), 0), reinterpret_tensor(buf87, (256, 144, 32), (4608, 32, 1), 0), out=buf88)
        del buf87
        buf89 = reinterpret_tensor(buf67, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf67  # reuse
        # Source Nodes: [x_162], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf88, arg153_1, arg154_1, arg46_1, arg47_1, buf89, 524288, grid=grid(524288), stream=stream0)
        del arg153_1
        del arg154_1
        del arg46_1
        del arg47_1
        buf90 = reinterpret_tensor(buf88, (8, 256, 16, 16), (65536, 256, 16, 1), 0); del buf88  # reuse
        # Source Nodes: [x_165, x_166], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_26.run(buf89, buf90, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del buf89
        # Source Nodes: [x_165, x_166], Original ATen: [aten.convolution, aten.silu]
        buf91 = extern_kernels.convolution(buf90, arg97_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg97_1
        del buf90
        buf92 = buf71; del buf71  # reuse
        buf93 = buf92; del buf92  # reuse
        # Source Nodes: [shortcut_6, x_167, x_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_27.run(buf93, buf91, arg155_1, arg156_1, arg48_1, arg49_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg155_1
        del arg156_1
        del arg48_1
        del arg49_1
        del buf91
        # Source Nodes: [x_174], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg98_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg98_1
        buf95 = buf94; del buf94  # reuse
        buf96 = buf95; del buf95  # reuse
        # Source Nodes: [x_175, x_179], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf96, arg157_1, arg158_1, arg50_1, arg51_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg157_1
        del arg158_1
        del arg50_1
        del arg51_1
        # Source Nodes: [kv_3], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, arg100_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 640, 16, 16), (163840, 256, 16, 1))
        del arg100_1
        # Source Nodes: [q_5], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf96, arg99_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 128, 8, 8), (8192, 64, 8, 1))
        del arg99_1
        del buf96
        buf99 = empty((64, 4, 16, 16), device='cuda', dtype=torch.float32)
        buf104 = empty((64, 4, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4, q_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf98, buf99, buf104, 4096, 16, grid=grid(4096, 16), stream=stream0)
        buf100 = buf78; del buf78  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf97, buf100, 589824, grid=grid(589824), stream=stream0)
        buf101 = empty((256, 16, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf99, (256, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf100, (256, 16, 144), (2304, 144, 1), 0), out=buf101)
        buf102 = reinterpret_tensor(buf99, (256, 4, 4, 16), (256, 64, 16, 1), 0); del buf99  # reuse
        # Source Nodes: [x_185], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf98, buf102, 1024, 64, grid=grid(1024, 64), stream=stream0)
        del buf98
        buf103 = empty((4096, 23), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_185], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (4096, 16), (16, 1), 0), reinterpret_tensor(arg53_1, (16, 23), (1, 16), 0), out=buf103)
        del arg53_1
        del buf102
        buf105 = empty((4096, 23), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_181], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (4096, 16), (16, 1), 0), reinterpret_tensor(arg52_1, (16, 23), (1, 16), 0), out=buf105)
        del arg52_1
        buf108 = buf100; del buf100  # reuse
        # Source Nodes: [attn_2, attn_3, mul_6], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_32.run(buf101, buf103, buf105, buf108, 4096, 144, grid=grid(4096), stream=stream0)
        buf109 = reinterpret_tensor(buf86, (64, 4, 144, 64), (36864, 9216, 64, 1), 0); del buf86  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf97, buf109, 36864, 64, grid=grid(36864, 64), stream=stream0)
        del buf97
        buf110 = reinterpret_tensor(buf82, (256, 16, 64), (1024, 64, 1), 0); del buf82  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf108, (256, 16, 144), (2304, 144, 1), 0), reinterpret_tensor(buf109, (256, 144, 64), (9216, 64, 1), 0), out=buf110)
        del buf109
        buf111 = reinterpret_tensor(buf80, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf80  # reuse
        # Source Nodes: [x_190], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_34.run(buf110, arg159_1, arg160_1, arg54_1, arg55_1, buf111, 262144, grid=grid(262144), stream=stream0)
        del arg159_1
        del arg160_1
        del arg54_1
        del arg55_1
        buf112 = reinterpret_tensor(buf110, (8, 512, 8, 8), (32768, 64, 8, 1), 0); del buf110  # reuse
        # Source Nodes: [x_193, x_194], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf111, buf112, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del buf111
        # Source Nodes: [x_193, x_194], Original ATen: [aten.convolution, aten.silu]
        buf113 = extern_kernels.convolution(buf112, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 2048, 8, 8), (131072, 64, 8, 1))
        del arg101_1
        # Source Nodes: [x_201], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf93, arg102_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 2048, 8, 8), (131072, 64, 8, 1))
        del arg102_1
        del buf93
        buf115 = buf113; del buf113  # reuse
        buf116 = buf115; del buf115  # reuse
        # Source Nodes: [shortcut_7, x_195, x_202, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_silu_36.run(buf116, arg161_1, arg162_1, arg56_1, arg57_1, buf114, arg163_1, arg164_1, arg58_1, arg59_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg161_1
        del arg162_1
        del arg163_1
        del arg164_1
        del arg56_1
        del arg57_1
        del arg58_1
        del arg59_1
        del buf114
        # Source Nodes: [x_207], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, arg103_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (8, 512, 8, 8), (32768, 64, 8, 1))
        del arg103_1
        buf118 = buf117; del buf117  # reuse
        buf119 = buf118; del buf118  # reuse
        # Source Nodes: [x_208, x_212], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_37.run(buf119, arg165_1, arg166_1, arg60_1, arg61_1, 262144, grid=grid(262144), stream=stream0)
        del arg165_1
        del arg166_1
        del arg60_1
        del arg61_1
        # Source Nodes: [kv_6], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg105_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 640, 8, 8), (40960, 64, 8, 1))
        del arg105_1
        # Source Nodes: [q_10], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf119, arg104_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (8, 128, 8, 8), (8192, 64, 8, 1))
        del arg104_1
        buf122 = empty((8, 640, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [kv_7], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_38.run(buf120, buf122, 737280, grid=grid(737280), stream=stream0)
        del buf120
        buf123 = empty((64, 16, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        triton_poi_fused_bmm_39.run(buf122, buf123, 147456, grid=grid(147456), stream=stream0)
        buf124 = reinterpret_tensor(buf108, (64, 64, 144), (9216, 144, 1), 0); del buf108  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf121, (64, 64, 16), (1024, 1, 64), 0), buf123, out=buf124)
        del buf123
        buf125 = reinterpret_tensor(buf104, (64, 8, 8, 16), (1024, 128, 16, 1), 0); del buf104  # reuse
        # Source Nodes: [x_218], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf121, buf125, 512, 128, grid=grid(512, 128), stream=stream0)
        buf126 = buf105; del buf105  # reuse
        # Source Nodes: [x_218], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (4096, 16), (16, 1), 0), reinterpret_tensor(arg63_1, (16, 23), (1, 16), 0), out=buf126)
        del arg63_1
        buf127 = buf125; del buf125  # reuse
        # Source Nodes: [x_214], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf121, buf127, 4096, 16, grid=grid(4096, 16), stream=stream0)
        del buf121
        buf128 = buf103; del buf103  # reuse
        # Source Nodes: [x_214], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (4096, 16), (16, 1), 0), reinterpret_tensor(arg62_1, (16, 23), (1, 16), 0), out=buf128)
        del arg62_1
        del buf127
        buf131 = reinterpret_tensor(buf101, (64, 1, 64, 144), (9216, 9216, 144, 1), 0); del buf101  # reuse
        # Source Nodes: [attn_4, attn_5, mul_7], Original ATen: [aten._softmax, aten.add, aten.mul]
        triton_per_fused__softmax_add_mul_42.run(buf124, buf126, buf128, buf131, 4096, 144, grid=grid(4096), stream=stream0)
        del buf126
        del buf128
        buf132 = reinterpret_tensor(buf124, (64, 144, 64), (9216, 1, 144), 0); del buf124  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        triton_poi_fused_bmm_43.run(buf122, buf132, 589824, grid=grid(589824), stream=stream0)
        del buf122
        buf133 = reinterpret_tensor(buf119, (64, 64, 64), (4096, 64, 1), 0); del buf119  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf131, (64, 64, 144), (9216, 144, 1), 0), buf132, out=buf133)
        del buf131
        del buf132
        buf134 = reinterpret_tensor(buf112, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf112  # reuse
        # Source Nodes: [x_223], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_44.run(buf133, arg167_1, arg168_1, arg64_1, arg65_1, buf134, 262144, grid=grid(262144), stream=stream0)
        del arg167_1
        del arg168_1
        del arg64_1
        del arg65_1
        buf135 = reinterpret_tensor(buf133, (8, 512, 8, 8), (32768, 64, 8, 1), 0); del buf133  # reuse
        # Source Nodes: [x_226, x_227], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf134, buf135, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del buf134
        # Source Nodes: [x_226, x_227], Original ATen: [aten.convolution, aten.silu]
        buf136 = extern_kernels.convolution(buf135, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 2048, 8, 8), (131072, 64, 8, 1))
        del arg106_1
        del buf135
        buf137 = buf116; del buf116  # reuse
        buf138 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cuda', dtype=torch.float32)
        buf139 = reinterpret_tensor(buf138, (8, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf138  # reuse
        # Source Nodes: [x_228, x_234, x_235, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_silu_45.run(buf137, buf139, buf136, arg169_1, arg170_1, arg66_1, arg67_1, 16384, 64, grid=grid(16384), stream=stream0)
        del arg169_1
        del arg170_1
        del arg66_1
        del arg67_1
        del buf136
        del buf137
        buf140 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_242], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg108_1, reinterpret_tensor(buf139, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg107_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf140)
        del arg107_1
        del arg108_1
        return (buf140, )


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
    arg44_1 = rand_strided((23, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((23, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((23, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((23, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((23, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((23, 16), (16, 1), device='cuda:0', dtype=torch.float32)
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
    arg95_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((384, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((640, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((640, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('eca_halonext26ts', benchmark_compiled_module)
