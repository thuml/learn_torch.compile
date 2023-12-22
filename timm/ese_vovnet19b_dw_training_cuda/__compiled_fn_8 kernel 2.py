
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


# kernel path: /tmp/torchinductor_youkaichao/qt/cqtcbt4g57hrviadj7jcv3toxxh4xcnjikjw4nylbesjf4s5onol.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_0', 'mutated_arg_names': []}
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

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/ug/cugjhknx7nb5uliec2o63wjmwnqry5ufq5z4dktaea2ec6rejo3j.py
# Source Nodes: [], Original ATen: [aten.div, aten.hardsigmoid_backward, aten.mul, aten.sum]

triton_per_fused_div_hardsigmoid_backward_mul_sum_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_hardsigmoid_backward_mul_sum_1', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x3 = xindex
    r2 = rindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0 + (1024*r2) + (50176*x1)), rmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last').to(tl.int1)
    tmp1 = 49.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp10 = 0.16666666666666666
    tmp11 = tmp8 * tmp10
    tmp12 = 0.0
    tmp13 = tl.where(tmp9, tmp11, tmp12)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/w2/cw2zkej6ssh7zwlvkwtre2sfwp35thyodryteta4pzdhgbx4jrl6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/camwu2ldxnw54xynmm2iphevhitqfhzn67jureq2eeafmxtioida.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (1024*(r2 // 49)) + (2048*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (1024*(r2 // 49)) + (2048*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (1024*(r2 // 49)) + (2048*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr4 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = 49.0
        tmp5 = tmp3 / tmp4
        tmp7 = tmp5 * tmp6
        tmp9 = tmp8 / tmp4
        tmp10 = tmp7 + tmp9
        tmp11 = tl.where(tmp2, tmp1, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp11 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask, tmp21, _tmp20)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, None)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yi/cyisw5bfv7sttadcmwaaiyzrb6vtf4i5q5d5vcx6elpdkghooon6.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zp/czpqicojc537bsxuebnyavinbqa37r4nkkru6bdasvtdnw7sbonk.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xl/cxlzavlxxrw7jm6koyi4v7kf72bxvcdkox4dnpdtd6njkrsguvuc.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1024
    x2 = (xindex // 50176)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x3), None)
    tmp13 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = 49.0
    tmp5 = tmp3 / tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8 / tmp4
    tmp10 = tmp7 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp14 = tmp12 - tmp13
    tmp16 = 0.002551020408163265
    tmp17 = tmp15 * tmp16
    tmp19 = tmp18 * tmp18
    tmp20 = tmp17 * tmp19
    tmp21 = tmp14 * tmp20
    tmp22 = tmp11 - tmp21
    tmp24 = tmp23 * tmp16
    tmp25 = tmp22 - tmp24
    tmp27 = tmp18 * tmp26
    tmp28 = tmp25 * tmp27
    tl.store(in_out_ptr0 + (x3), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5dlrx7xltlunu62i2zamye4r7jkrnives4vjlklv2quykbrwfi.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 224
    x1 = (xindex // 224)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (224*r2) + (21952*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (59584 + (49*x0) + (70560*(r2 // 49)) + (141120*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (224*r2) + (21952*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ta/ctalqozayq3pa4igprhzbw75hojomurupflcst6zov3zdneamdgc.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (224*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5i/c5iowrthqmij52lsdq2jyygu3vshpiqv3drpgx6drfftazq7fn3s.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (224*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ie/cie6vdm277koguogk4tlyy3nlervv6zfoo43nrhqd2xk5mjcy2zn.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 224
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
    tmp0 = tl.load(in_ptr0 + (x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (59584 + y0 + (49*x2) + (70560*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.002551020408163265
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x2 + (224*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5p/c5pxr4uktg5zcitwko7pu4jwaxozgz4ky6y3tgjovwxlywf573hm.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
    xnumel = 224
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex
    r1 = rindex % 49
    r2 = (rindex // 49)
    tmp0 = tl.load(in_ptr0 + (x0 + (224*r3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (48608 + r1 + (49*x0) + (70560*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dh/cdhmzwi7x4pcqyb2rencrntlfhjc5dygstlnmw73rwt736lst2hw.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 224
    x1 = (xindex // 224)
    tmp8 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (224*r2) + (21952*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (48608 + (49*x0) + (70560*(r2 // 49)) + (141120*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((49*x0) + (10976*(r2 // 49)) + (21952*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x0 + (224*r2) + (21952*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ff/cff7v4skn3zbirezhc7vdgggmu6ekazfvnxgeauwjws4nxkx4e2f.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 224
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
    tmp0 = tl.load(in_ptr0 + (x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (48608 + y0 + (49*x2) + (70560*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (49*x2) + (10976*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.002551020408163265
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2 + (224*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nx/cnxuhlwx7jqhmis34mdd73glohq5mt4eoluwhmhbgorrwtjgmeah.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
    xnumel = 224
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex
    r1 = rindex % 49
    r2 = (rindex // 49)
    tmp0 = tl.load(in_ptr0 + (x0 + (224*r3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (37632 + r1 + (49*x0) + (70560*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v5/cv52uxn224m7v37nu74s5azrozy2klwo5rtkjbw3ey4hlbnx6hpw.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 224
    x1 = (xindex // 224)
    tmp8 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (224*r2) + (21952*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (37632 + (49*x0) + (70560*(r2 // 49)) + (141120*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((49*x0) + (10976*(r2 // 49)) + (21952*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x0 + (224*r2) + (21952*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q4/cq4eqag6p5o3maf2dbi6amsn7vxdzulolrcsszey56q4ruf7sctg.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 224
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
    tmp0 = tl.load(in_ptr0 + (x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (37632 + y0 + (49*x2) + (70560*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (49*x2) + (10976*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.002551020408163265
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2 + (224*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ro/crophvajz35yyrn2xqtlmkt7mybe7cy3ov3gxafrkvjm4dp57khy.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 224
    x1 = (xindex // 224)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (224*r2) + (21952*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((49*x0) + (10976*(r2 // 49)) + (21952*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (224*r2) + (21952*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7lyxxyge6twxdkavc5oglcdeybajcnyllimjkwnpu5v3tz2qsoi.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 224
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
    tmp0 = tl.load(in_ptr0 + (x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (49*x2) + (10976*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.002551020408163265
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (224*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xk/cxk2er3vtv2ql2fdrwuznlsf63smdtwxst2rzqbw4cozybf6vvtd.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 37632
    x1 = (xindex // 37632)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (70560*x1)), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nx/cnxfyrfiqhuwpslk7ijavenvfl2ktl5z5cficujgjm2tawoolyd3.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
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
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (75264*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (75264*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/td/ctdq2vanvewdq4bvtedxw7m5vm4x52uvfksuxgulri6bjykgyn7t.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_mul_sum_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_mul_sum_21', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 768
    x1 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (1536*x1)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.16666666666666666
    tmp7 = tmp4 * tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5r/c5rat36i7dfmz7upv2p6khp5kbpx36prbrkedlpve3twsygxkox3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2s/c2snxozgf3fistvks6myfd345ro5mipraykizen2ihinu2kulfbl.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9984
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 768)
    x0 = xindex % 768
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (768*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (768*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (768*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = tl.load(in_ptr3 + (x0 + (768*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = 196.0
        tmp11 = tmp9 / tmp10
        tmp12 = tmp8 + tmp11
        tmp13 = tl.where(tmp5, tmp4, tmp12)
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp19 = tl.load(in_ptr4 + (x0 + (768*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp13 * tmp21
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mb/cmb7t6ouuzzc6rgtjhjnxn66so6tquezcdtcdxomorvqwjbs5ln6.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/ao/caoas7sc5zjlpj3dqfj3qnrb7q4t6optum3yhtv6y5qkisgk6bg7.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ff/cffvcnna2n2y5liq6aauwfixra5zimuwr66k2ka53yh2qtzoyghy.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 768
    x2 = (xindex // 150528)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x0 + (768*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0 + (768*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x3), None)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 196.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.0006377551020408163
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(in_out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qf/cqfeyrhx57hbszptoaz4ggkz4wy7dgywzjd5np756bk77cup6ec2.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2496
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (192*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (175616 + (196*x1) + (213248*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sw/csw4lygmuj5jl45so6fq3jdiew3cvqw4t2ivu2xq4c5kljgue6cv.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
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


# kernel path: /tmp/torchinductor_youkaichao/oy/coy4ix3n4e2gtgyn6cekl5iprqzwbfsqthj2cv4fzbywuq6s7xjs.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2496
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (175616 + (196*x0) + (213248*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.load(in_ptr2 + (x0 + (192*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f5/cf5dtjx4526psa3tsrifyulmo2k4kiiity3wndub5miqqcy34l2x.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coecicculrltjyxn32jpoeft3gemfoowaslcctaarb6myrpx666k.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (175616 + y0 + (196*x2) + (213248*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.0006377551020408163
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/db/cdbnghnjkr543nyu5ol2c4ipnvk23igkujgzckdz5c2mf4wnt43v.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (137984 + r1 + (196*x0) + (213248*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7q/c7qkfm2bmdmowa4ugvqtrfokbzu5ci22e7indo7p4ko2kf5iyds4.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2496
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (192*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (137984 + (196*x1) + (213248*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + ((196*x1) + (37632*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.where(tmp5, tmp4, tmp8)
        tmp10 = tl.load(in_ptr3 + (x1 + (192*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr4 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp9 * tmp12
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dn/cdndv7qyissekjnhglofvnud2jd6x5c6arwvsf4lyerxyadqyiff.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ci/cci6z6xe34p5ehk3tmgt4hnphuwl7vstiwss2ucaox5tus3ngvx2.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (137984 + y0 + (196*x2) + (213248*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (196*x2) + (37632*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.0006377551020408163
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y5/cy5s67e37blexphye5x3ywvigsnj6d7s5g5kirocsn4ypir4aaji.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (100352 + r1 + (196*x0) + (213248*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (37632*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4a/c4a2dnihh2vkdl5p3tw7ttqzkbvntr3vtde6ea75gji2xx545sr5.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2496
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (192*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (100352 + (196*x1) + (213248*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + ((196*x1) + (37632*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.where(tmp5, tmp4, tmp8)
        tmp10 = tl.load(in_ptr3 + (x1 + (192*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr4 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp9 * tmp12
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ug/cug2adfhyyqmzegxjfaoy5dkovto6yzdf5d2f3eodpxzardybdbe.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (100352 + y0 + (196*x2) + (213248*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (196*x2) + (37632*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.0006377551020408163
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7v/c7vxvhemse3lxe4rms4gg6nsf2ug6bad64eq4hd4slnmaaovekpi.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2496
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (192*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((196*x1) + (37632*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wi/cwiun2iybb5cvba5i5pwpwv2gt3z7322in435a3impz2fb5r34xm.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2496
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((196*x0) + (37632*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (192*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hx/chxe4nuzgvop4a6kxxfjxwh4wtn4onueoarnwglgtwlcwnkptkbm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (196*x2) + (37632*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5a/c5ahintlb6ljrchixhv2nqkyguevpkzvfufnzyjc55l4sdstq3x3.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 100352
    x1 = (xindex // 100352)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (213248*x1)), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dl/cdlt4itoklw2hwbi4hvbzm6bgkeac5k27ivuwgkqsoe6xkftz3ep.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 28672
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (57344*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (57344*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ds/cdsb7cpux7i2v4mzhrepok3drxsxnbrbforjkyro34zqbmhcqttd.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_mul_sum_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_mul_sum_44', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (3584*x1)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.16666666666666666
    tmp7 = tmp4 * tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yg/cygb5jvt3myocwhyj4bpzcgkv2jlhf5shvh5foq2qbrpmpzv4acm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ic/cicpeixguttdlbxgzw7p6oe4uu3zzcq4w66jx6matp36lkeegbva.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (512*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x0 + (512*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr4 + (x0 + (512*r2) + (65536*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = 784.0
        tmp8 = tmp6 / tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hd/chduprft2473c7g4s6wfvo4antzcaxge5skbtm6fp5zfrpcetitx.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ck/cckdfbbrh5nrjnad36f4n65o4j2owqrrcwmfz2byunsnst6ttgoq.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ts/ctse2n5rdx343dfbcp3qggnkfnn7ejhgjk3oiau5yyvnbfif73dh.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 512
    x2 = (xindex // 401408)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x3), None)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 784.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.00015943877551020407
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(in_out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gh/cgh2lmw3m2dzhcsvswahhl3fodjie5rfm4tz6yyrpipr2lpil2j6.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7840
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (160*r2) + (20480*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (451584 + (784*x1) + (577024*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xx/cxxomct3smyfp56oeqhr7exxz7mxnaycdqm4deukjy4s4axm6v6x.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
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


# kernel path: /tmp/torchinductor_youkaichao/gf/cgfm6lihfz2vwc6ul5atwlaok6dm3kvu4rjjc4melme4efeimqm7.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7840
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 160
    x1 = (xindex // 160)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (160*r2) + (20480*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (451584 + (784*x0) + (577024*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (160*r2) + (20480*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5b/c5bjv2l4fedgymkudvzqexfnrjwgmtsgm55rze4fhpf26bd74bup.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (160*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pe/cpezktmymzlikxse2x7xt5nkzh7v75prya2ialxtcw7lar7vumgg.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 160
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
    tmp0 = tl.load(in_ptr0 + (x2 + (160*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (451584 + y0 + (784*x2) + (577024*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (160*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 0.00015943877551020407
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x2 + (160*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lv/clvqiuvsigk4xvzytuxj66rpxkxghn2pooesk3pn6cvlrl42nrnx.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (x0 + (160*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (326144 + r1 + (784*x0) + (577024*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ua/cuauiwk5wcz5ey4dtg66mjkhhvth7ijhqyzwwtce2qx7tafphmx5.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7840
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (160*r2) + (20480*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (326144 + (784*x1) + (577024*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((784*x1) + (125440*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (160*r2) + (20480*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fw/cfwmxem37k5abdua6sqo2mk2bbz5aekwalktryi56vwkwmfscl2p.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zp/czpjimlwxlhfdysaovkgu25qlrf3pbtpdm5amsan3skmf635bx2r.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 160
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
    tmp0 = tl.load(in_ptr0 + (x2 + (160*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (326144 + y0 + (784*x2) + (577024*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (784*x2) + (125440*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (160*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.00015943877551020407
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2 + (160*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zn/cznbplwbuqa5vnnonfm5oxazhc6lofxvr2lwmnlt7mab6oabbrtn.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (x0 + (160*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (200704 + r1 + (784*x0) + (577024*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (125440*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f5/cf5fifuk2j3vrppvmtjlse44aasafew3kc7bsjorfajsw65ykvyg.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7840
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (160*r2) + (20480*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (200704 + (784*x1) + (577024*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((784*x1) + (125440*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (160*r2) + (20480*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jg/cjgfvoo42pmtkt4tc5h7nrthvtjkggqeacytlin6vs7wgsnlu25z.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 160
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
    tmp0 = tl.load(in_ptr0 + (x2 + (160*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (200704 + y0 + (784*x2) + (577024*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (784*x2) + (125440*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (160*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.00015943877551020407
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2 + (160*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ju/cjuouzk3w7lrhlhgdb4ueldyhnbev5mlyfzx7jmhovyvfovz2p3g.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7840
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (160*r2) + (20480*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((784*x1) + (125440*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w2/cw26nclfo6i2oenvjxowkm4nc3p4hfxtbyen6jczguruioiloac6.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_63 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7840
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 160
    x1 = (xindex // 160)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (160*r2) + (20480*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((784*x0) + (125440*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (160*r2) + (20480*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hg/chgj7zjolzpsq6qf54d33ikmsqp75bgyt2z42dgup6w2q2fty2ta.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 160
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
    tmp0 = tl.load(in_ptr0 + (x2 + (160*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (125440*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (160*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00015943877551020407
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (160*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5t/c5t2btaufnnvg7gprehwhlayuekz65qkh4ogan5wrus2drcwa46g.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_65', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 200704
    x1 = (xindex // 200704)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (577024*x1)), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/va/cvaz3mjfzjvkcuh6s36ainwikcsxyvd3wo5z4qfe35tv4cw2i4xa.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 51200
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 256) % 25
    x0 = xindex % 256
    x2 = (xindex // 6400)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*((r3 + (126*x1)) % 3136)) + (802816*x2)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (256*((r3 + (126*x1)) % 3136)) + (802816*x2)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fa/cfajjkm74xlk3zxtpeb5ssffqkaxp3qtk4chtvp4jlp5cbe5kdsz.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_mul_sum_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 32],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_mul_sum_67', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (6400*x1)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.16666666666666666
    tmp7 = tmp4 * tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/p2/cp2ni57wnugy5uh7xtzonpu3yh4vkr57z32syz23wcizh4r2qyxw.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rq/crq4dag5ba2e2w4yhaamivach2dz3wv5qjdjylbqqpkx7hd77b2q.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (256*((r2 + (128*x1)) // 3136))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x0 + (256*((r2 + (128*x1)) // 3136))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr4 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = 3136.0
        tmp8 = tmp6 / tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hb/chbfktqxez2fswhh77iik2davoam7pql3da54lyq4gxgos36e6un.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_70 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
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
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lt/cltirqh7qgox56qnjslssv42omke6tvytlypbhulwszpz4lfyfku.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
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
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xr/cxrxsdhjr7jouf7vprgnxydbfkezevjjpohbigz3ou2d4g6rm73o.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_72', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 256
    x2 = (xindex // 802816)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x3), None)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 3136.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 3.985969387755102e-05
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.store(in_out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ar/care4yqbgfqe3opjg73p4tmeqil2g4wuvj3yetfqywfivsd2wiw6.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (1003520 + (3136*x1) + (1404928*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/caejccecuz2bffadij23qwsh66cseb4dknzsjspz5lxp7vykioyd.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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


# kernel path: /tmp/torchinductor_youkaichao/lp/clpvkdph3nsrsatnp4b4c3y3zrtqfmpgsmsvfcjq2cnaul5iifom.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (1003520 + (3136*x0) + (1404928*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nd/cndnue7n5nhrwpsl2digtxnznqy6pax33u2vojp7atgwgou7dhfy.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hk/chk2lnbxaivhbxydudpybqqdmswktragvljggml5tezofseeijyx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (1003520 + y0 + (3136*x2) + (1404928*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 3.985969387755102e-05
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hi/chi5kxo5ypdsemletwlcw5qxgj2lqgbesirwcsclrupb6skpgjwi.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (802816*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (602112 + (3136*x0) + (1404928*(r2 // 3136)) + (2809856*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zs/czs7rw2eoj3ym3q3llzqxyb4gfbufptytqin4pqfmgycqtvh2uhv.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_79', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fe/cfein2umlgswvr55vfmjed3nxxtskf2mt3a3pd4c3t564wzvsht3.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (602112 + (3136*x1) + (1404928*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x1) + (401408*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tt/cttrskb2k4kj5hjzajv4vuwodvdbbfoz4e3dzqvcs5mc4osyr7gy.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jk/cjkocz2wwq3cqhs43zoaq2augtaumubiccnz3nfho5lkr7quj6ns.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (602112 + y0 + (3136*x2) + (1404928*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (3136*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 3.985969387755102e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zg/czg2sg7jn2e2326z7bwp3jldu6b3tjiu42aisarx6zx4cqxvsa57.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_83', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (802816*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (200704 + (3136*x0) + (1404928*(r2 // 3136)) + (2809856*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wx/cwx2twdiua7cgltppikimhiaxxempr3axwyz4sgj2jlx2gsqmuvi.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_84', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (200704 + (3136*x1) + (1404928*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x1) + (401408*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/db/cdbnobqhb3wl3ks6zd64hopnejqr6iwmmlt3ky4pyw5fe3buyael.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_85 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_85', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (200704 + y0 + (3136*x2) + (1404928*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (3136*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 3.985969387755102e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vs/cvssfulbeffcjw2zvg6ztxoupm7xhq72u4vdu2kl2fsg2s54iaz6.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_86', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x1) + (401408*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2p/c2pbfz5dqvufjwjnbvq2qqzg7wxf3ysymqbvs5qyin3rq3uvoa4u.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (401408*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/co/ccocmc5en6qaqnnnwptzefpu5ah2vzgqdi47mfb6tmsdr5liromm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_88 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 3.985969387755102e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b3/cb3xkfwne2wjhbtprd5ifxkdfzd7p6zju6g3b3jrrmqasoywu3xl.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_89 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (1404928*(r2 // 3136)) + (2809856*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gv/cgvo6wlnae5po2onhohfr2kndtc5c7uz3tvrplv7iuhwpy365axu.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_90 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j3/cj3ser5smtah4k3srcaeyv2rhvipg2zpgcmrxwtlnioiug4wnmqp.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x1) + (1404928*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x1) + (200704*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xs/cxsvny5q3uqnud64ly6r6gcgjfem6p7gmzj6akkrimurhie52uw5.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_92 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_92', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pb/cpbl7kdl4kutlfkt2s2no36f5y47w4wad3mjjvjxh6xk5hfv6ooq.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_93 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (1404928*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (3136*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 3.985969387755102e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pa/cpaxnzcnjfdh53bgk7dagixfsspsoaccdtllmtbqodwaiilmnyws.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_94 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_94', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x1) + (802816*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6c/c6ccj4id2ivn7stfspfv3idnfwigd5nii25e2komgoyjkze5edho.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_95 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 64
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


# kernel path: /tmp/torchinductor_youkaichao/rd/crdrmshrzfn77gb3orkx6e5y4dr7aufjae37ozizpt3hk4yklugz.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_96 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_96', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x0) + (802816*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6b/c6btwhwe23ylx6mzyq2uxdx6zgyua2iap7njif34324qqpoxu6gg.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_97 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 784
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u6/cu62ggpj5i4ailcxeqtnkal3edtcokey3aqngz7m4bzarahz2eg6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_98 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_98', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100352
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 12544
    y1 = (yindex // 12544)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (12544*x2) + (802816*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 9.964923469387754e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp21, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_163, convolution, squeeze_1, relu, convolution_1, convolution_2, squeeze_4, relu_1, convolution_3, convolution_4, squeeze_7, relu_2, convolution_5, squeeze_10, relu_3, convolution_6, convolution_7, squeeze_13, relu_4, convolution_8, convolution_9, squeeze_16, relu_5, convolution_10, convolution_11, squeeze_19, cat, convolution_12, squeeze_22, relu_7, mean, div, mul_56, getitem_16, getitem_17, convolution_14, squeeze_25, relu_8, convolution_15, convolution_16, squeeze_28, relu_9, convolution_17, convolution_18, squeeze_31, relu_10, convolution_19, convolution_20, squeeze_34, cat_1, convolution_21, squeeze_37, relu_12, mean_1, div_1, mul_92, getitem_28, getitem_29, convolution_23, squeeze_40, relu_13, convolution_24, convolution_25, squeeze_43, relu_14, convolution_26, convolution_27, squeeze_46, relu_15, convolution_28, convolution_29, squeeze_49, cat_2, convolution_30, squeeze_52, relu_17, mean_2, div_2, mul_128, getitem_40, getitem_41, convolution_32, squeeze_55, relu_18, convolution_33, convolution_34, squeeze_58, relu_19, convolution_35, convolution_36, squeeze_61, relu_20, convolution_37, convolution_38, squeeze_64, cat_3, convolution_39, squeeze_67, relu_22, mean_3, div_3, clone, permute_1, bitwise_and, unsqueeze_94, le_1, unsqueeze_106, unsqueeze_118, unsqueeze_130, unsqueeze_142, bitwise_and_1, unsqueeze_154, le_6, unsqueeze_166, unsqueeze_178, unsqueeze_190, unsqueeze_202, bitwise_and_2, unsqueeze_214, le_11, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, bitwise_and_3, unsqueeze_274, le_16, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_17, (160, ), (1, ))
    assert_size_stride(primals_19, (160, ), (1, ))
    assert_size_stride(primals_21, (160, ), (1, ))
    assert_size_stride(primals_23, (160, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_27, (192, ), (1, ))
    assert_size_stride(primals_29, (192, ), (1, ))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_33, (192, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_37, (224, ), (1, ))
    assert_size_stride(primals_39, (224, ), (1, ))
    assert_size_stride(primals_41, (224, ), (1, ))
    assert_size_stride(primals_43, (224, ), (1, ))
    assert_size_stride(primals_45, (1024, ), (1, ))
    assert_size_stride(primals_47, (64, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_48, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_49, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_50, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_51, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_52, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_53, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_54, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_55, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_56, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_57, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_58, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_59, (256, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_60, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_62, (160, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_63, (160, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_64, (160, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_65, (160, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_66, (160, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_67, (160, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_68, (160, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_69, (512, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_70, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_72, (192, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_73, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_74, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_75, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_76, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_77, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_78, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_79, (768, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_80, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_82, (224, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_83, (224, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_84, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_85, (224, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_86, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_87, (224, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_88, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_89, (1024, 1440, 1, 1), (1440, 1, 1, 1))
    assert_size_stride(primals_90, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_163, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_1, (64, ), (1, ))
    assert_size_stride(relu, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(convolution_1, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(convolution_2, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_4, (64, ), (1, ))
    assert_size_stride(relu_1, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(convolution_3, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_4, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(relu_2, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_5, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_10, (128, ), (1, ))
    assert_size_stride(relu_3, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_6, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_7, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_13, (128, ), (1, ))
    assert_size_stride(relu_4, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_8, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_9, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_16, (128, ), (1, ))
    assert_size_stride(relu_5, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_10, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_11, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_19, (128, ), (1, ))
    assert_size_stride(cat, (8, 448, 56, 56), (1404928, 1, 25088, 448))
    assert_size_stride(convolution_12, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_22, (256, ), (1, ))
    assert_size_stride(relu_7, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(mean, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(div, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(mul_56, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(getitem_16, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(getitem_17, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_14, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(squeeze_25, (160, ), (1, ))
    assert_size_stride(relu_8, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(convolution_15, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(convolution_16, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(squeeze_28, (160, ), (1, ))
    assert_size_stride(relu_9, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(convolution_17, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(convolution_18, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(squeeze_31, (160, ), (1, ))
    assert_size_stride(relu_10, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(convolution_19, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(convolution_20, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(squeeze_34, (160, ), (1, ))
    assert_size_stride(cat_1, (8, 736, 28, 28), (577024, 1, 20608, 736))
    assert_size_stride(convolution_21, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_37, (512, ), (1, ))
    assert_size_stride(relu_12, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(mean_1, (8, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(div_1, (8, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(mul_92, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(getitem_28, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_29, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_23, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(squeeze_40, (192, ), (1, ))
    assert_size_stride(relu_13, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_24, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_25, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(squeeze_43, (192, ), (1, ))
    assert_size_stride(relu_14, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_26, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_27, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(squeeze_46, (192, ), (1, ))
    assert_size_stride(relu_15, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_28, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_29, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(squeeze_49, (192, ), (1, ))
    assert_size_stride(cat_2, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    assert_size_stride(convolution_30, (8, 768, 14, 14), (150528, 1, 10752, 768))
    assert_size_stride(squeeze_52, (768, ), (1, ))
    assert_size_stride(relu_17, (8, 768, 14, 14), (150528, 1, 10752, 768))
    assert_size_stride(mean_2, (8, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(div_2, (8, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(mul_128, (8, 768, 14, 14), (150528, 1, 10752, 768))
    assert_size_stride(getitem_40, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(getitem_41, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_32, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(squeeze_55, (224, ), (1, ))
    assert_size_stride(relu_18, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_33, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_34, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(squeeze_58, (224, ), (1, ))
    assert_size_stride(relu_19, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_35, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_36, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(squeeze_61, (224, ), (1, ))
    assert_size_stride(relu_20, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_37, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_38, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(squeeze_64, (224, ), (1, ))
    assert_size_stride(cat_3, (8, 1440, 7, 7), (70560, 1, 10080, 1440))
    assert_size_stride(convolution_39, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(squeeze_67, (1024, ), (1, ))
    assert_size_stride(relu_22, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(mean_3, (8, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(div_3, (8, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(clone, (8, 1024), (1024, 1))
    assert_size_stride(permute_1, (1000, 1024), (1024, 1))
    assert_size_stride(bitwise_and, (8, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(unsqueeze_94, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_1, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(unsqueeze_106, (1, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(unsqueeze_118, (1, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(unsqueeze_130, (1, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(unsqueeze_142, (1, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(bitwise_and_1, (8, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(unsqueeze_154, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(le_6, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(unsqueeze_166, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_178, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_190, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_202, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(bitwise_and_2, (8, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(unsqueeze_214, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_11, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(unsqueeze_226, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_238, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_250, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_262, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(bitwise_and_3, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(unsqueeze_274, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_16, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(unsqueeze_286, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_298, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_310, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_322, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_334, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_358, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone, out=buf1)
        del clone
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty_strided((8, 1024, 1, 1), (1024, 1, 8192, 8192), device='cuda', dtype=torch.float32)
        buf4 = reinterpret_tensor(buf3, (8, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf3  # reuse
        # Source Nodes: [], Original ATen: [aten.div, aten.hardsigmoid_backward, aten.mul, aten.sum]
        triton_per_fused_div_hardsigmoid_backward_mul_sum_1.run(buf4, buf0, relu_22, bitwise_and, 8192, 49, grid=grid(8192), stream=stream0)
        del bitwise_and
        buf5 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_2.run(buf4, buf5, 1024, 8, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf6 = aten.convolution_backward(buf4, mean_3, primals_90, [1024], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf4
        del mean_3
        del primals_90
        buf7 = buf6[0]
        buf8 = buf6[1]
        del buf6
        buf9 = empty_strided((1024, 4), (1, 1024), device='cuda', dtype=torch.float32)
        buf11 = empty_strided((1024, 4), (1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_3.run(relu_22, buf0, div_3, buf7, convolution_39, unsqueeze_94, buf9, buf11, 4096, 98, grid=grid(4096), stream=stream0)
        buf10 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_4.run(buf9, buf10, 1024, 4, grid=grid(1024), stream=stream0)
        del buf9
        buf12 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf14 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_5.run(buf11, squeeze_67, buf12, buf14, 1024, 4, grid=grid(1024), stream=stream0)
        buf13 = empty_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda', dtype=torch.float32)
        buf15 = buf13; del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_6.run(buf15, relu_22, buf0, div_3, buf7, convolution_39, unsqueeze_94, buf12, squeeze_67, buf10, primals_45, 401408, grid=grid(401408), stream=stream0)
        del buf0
        del buf12
        del buf7
        del convolution_39
        del div_3
        del primals_45
        del relu_22
        del squeeze_67
        del unsqueeze_94
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf16 = aten.convolution_backward(buf15, cat_3, primals_89, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf15
        del cat_3
        del primals_89
        buf17 = buf16[0]
        buf18 = buf16[1]
        del buf16
        buf19 = empty_strided((224, 4), (1, 224), device='cuda', dtype=torch.float32)
        buf21 = empty_strided((224, 4), (1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_7.run(le_1, buf17, convolution_38, unsqueeze_106, buf19, buf21, 896, 98, grid=grid(896), stream=stream0)
        buf20 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_8.run(buf19, buf20, 224, 4, grid=grid(224), stream=stream0)
        buf22 = empty((224, ), device='cuda', dtype=torch.float32)
        buf23 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_9.run(buf21, squeeze_64, buf22, buf23, 224, 4, grid=grid(224), stream=stream0)
        buf24 = empty_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_10.run(le_1, buf17, convolution_38, unsqueeze_106, buf22, squeeze_64, buf20, primals_43, buf24, 392, 224, grid=grid(392, 224), stream=stream0)
        del convolution_38
        del le_1
        del primals_43
        del squeeze_64
        del unsqueeze_106
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf25 = aten.convolution_backward(buf24, convolution_37, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf24
        del convolution_37
        del primals_88
        buf26 = buf25[0]
        buf27 = buf25[1]
        del buf25
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf28 = aten.convolution_backward(buf26, relu_20, primals_87, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 224, [True, True, False])
        del primals_87
        buf29 = buf28[0]
        buf30 = buf28[1]
        del buf28
        buf31 = buf22; del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_11.run(relu_20, buf17, buf29, buf31, 224, 392, grid=grid(224), stream=stream0)
        buf32 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_12.run(relu_20, buf17, buf29, convolution_36, unsqueeze_118, buf32, 896, 98, grid=grid(896), stream=stream0)
        buf33 = empty((224, ), device='cuda', dtype=torch.float32)
        buf35 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_9.run(buf32, squeeze_61, buf33, buf35, 224, 4, grid=grid(224), stream=stream0)
        buf34 = reinterpret_tensor(buf26, (8, 224, 7, 7), (10976, 1, 1568, 224), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_13.run(relu_20, buf17, buf29, convolution_36, unsqueeze_118, buf33, squeeze_61, buf31, primals_41, buf34, 392, 224, grid=grid(392, 224), stream=stream0)
        del buf29
        del convolution_36
        del primals_41
        del relu_20
        del squeeze_61
        del unsqueeze_118
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf36 = aten.convolution_backward(buf34, convolution_35, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf34
        del convolution_35
        del primals_86
        buf37 = buf36[0]
        buf38 = buf36[1]
        del buf36
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf39 = aten.convolution_backward(buf37, relu_19, primals_85, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 224, [True, True, False])
        del primals_85
        buf40 = buf39[0]
        buf41 = buf39[1]
        del buf39
        buf42 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_14.run(relu_19, buf17, buf40, buf42, 224, 392, grid=grid(224), stream=stream0)
        buf43 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_15.run(relu_19, buf17, buf40, convolution_34, unsqueeze_130, buf43, 896, 98, grid=grid(896), stream=stream0)
        buf44 = empty((224, ), device='cuda', dtype=torch.float32)
        buf46 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_9.run(buf43, squeeze_58, buf44, buf46, 224, 4, grid=grid(224), stream=stream0)
        buf45 = reinterpret_tensor(buf37, (8, 224, 7, 7), (10976, 1, 1568, 224), 0); del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_16.run(relu_19, buf17, buf40, convolution_34, unsqueeze_130, buf44, squeeze_58, buf42, primals_39, buf45, 392, 224, grid=grid(392, 224), stream=stream0)
        del buf40
        del convolution_34
        del primals_39
        del relu_19
        del squeeze_58
        del unsqueeze_130
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf47 = aten.convolution_backward(buf45, convolution_33, primals_84, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf45
        del convolution_33
        del primals_84
        buf48 = buf47[0]
        buf49 = buf47[1]
        del buf47
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf50 = aten.convolution_backward(buf48, relu_18, primals_83, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 224, [True, True, False])
        del primals_83
        buf51 = buf50[0]
        buf52 = buf50[1]
        del buf50
        buf53 = buf43; del buf43  # reuse
        buf55 = buf19; del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_17.run(relu_18, buf51, convolution_32, unsqueeze_142, buf53, buf55, 896, 98, grid=grid(896), stream=stream0)
        buf54 = buf44; del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_8.run(buf53, buf54, 224, 4, grid=grid(224), stream=stream0)
        del buf53
        buf56 = empty((224, ), device='cuda', dtype=torch.float32)
        buf57 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_9.run(buf55, squeeze_55, buf56, buf57, 224, 4, grid=grid(224), stream=stream0)
        del buf55
        buf58 = reinterpret_tensor(buf48, (8, 224, 7, 7), (10976, 1, 1568, 224), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_18.run(relu_18, buf51, convolution_32, unsqueeze_142, buf56, squeeze_55, buf54, primals_37, buf58, 392, 224, grid=grid(392, 224), stream=stream0)
        del buf51
        del buf56
        del convolution_32
        del primals_37
        del relu_18
        del squeeze_55
        del unsqueeze_142
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf59 = aten.convolution_backward(buf58, getitem_40, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf58
        del getitem_40
        del primals_82
        buf60 = buf59[0]
        buf61 = buf59[1]
        del buf59
        buf62 = buf60; del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_19.run(buf62, buf17, 301056, grid=grid(301056), stream=stream0)
        del buf17
        # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
        buf63 = aten.max_pool2d_with_indices_backward(buf62, mul_128, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_41)
        del getitem_41
        del mul_128
        buf64 = buf63
        del buf63
        buf65 = empty_strided((8, 768, 1, 1, 2), (1536, 1, 12288, 12288, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_20.run(buf64, relu_17, buf65, 12288, 98, grid=grid(12288), stream=stream0)
        buf66 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cuda', dtype=torch.float32)
        buf67 = reinterpret_tensor(buf66, (8, 768, 1, 1), (768, 1, 768, 768), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_mul_sum_21.run(buf67, buf65, bitwise_and_1, 6144, 2, grid=grid(6144), stream=stream0)
        del bitwise_and_1
        del buf65
        buf68 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_22.run(buf67, buf68, 768, 8, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf69 = aten.convolution_backward(buf67, mean_2, primals_80, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf67
        del mean_2
        del primals_80
        buf70 = buf69[0]
        buf71 = buf69[1]
        del buf69
        buf72 = empty_strided((768, 13), (1, 768), device='cuda', dtype=torch.float32)
        buf74 = empty_strided((768, 13), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_23.run(relu_17, buf64, div_2, buf70, convolution_30, unsqueeze_154, buf72, buf74, 9984, 121, grid=grid(9984), stream=stream0)
        buf73 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_24.run(buf72, buf73, 768, 13, grid=grid(768), stream=stream0)
        del buf72
        buf75 = empty((768, ), device='cuda', dtype=torch.float32)
        buf77 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_25.run(buf74, squeeze_52, buf75, buf77, 768, 13, grid=grid(768), stream=stream0)
        del buf74
        buf76 = buf64; del buf64  # reuse
        buf78 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_26.run(buf78, relu_17, div_2, buf70, convolution_30, unsqueeze_154, buf75, squeeze_52, buf73, primals_35, 1204224, grid=grid(1204224), stream=stream0)
        del buf70
        del buf75
        del convolution_30
        del div_2
        del primals_35
        del relu_17
        del squeeze_52
        del unsqueeze_154
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf79 = aten.convolution_backward(buf78, cat_2, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf78
        del cat_2
        del primals_79
        buf80 = buf79[0]
        buf81 = buf79[1]
        del buf79
        buf82 = empty((192, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_27.run(le_6, buf80, buf82, 2496, 121, grid=grid(2496), stream=stream0)
        buf83 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_28.run(buf82, buf83, 192, 13, grid=grid(192), stream=stream0)
        buf84 = reinterpret_tensor(buf82, (192, 13), (1, 192), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_29.run(le_6, buf80, convolution_29, unsqueeze_166, buf84, 2496, 121, grid=grid(2496), stream=stream0)
        buf85 = empty((192, ), device='cuda', dtype=torch.float32)
        buf86 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_30.run(buf84, squeeze_49, buf85, buf86, 192, 13, grid=grid(192), stream=stream0)
        buf87 = reinterpret_tensor(buf62, (8, 192, 14, 14), (37632, 1, 2688, 192), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_31.run(le_6, buf80, convolution_29, unsqueeze_166, buf85, squeeze_49, buf83, primals_33, buf87, 1568, 192, grid=grid(1568, 192), stream=stream0)
        del convolution_29
        del le_6
        del primals_33
        del squeeze_49
        del unsqueeze_166
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf88 = aten.convolution_backward(buf87, convolution_28, primals_78, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf87
        del convolution_28
        del primals_78
        buf89 = buf88[0]
        buf90 = buf88[1]
        del buf88
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf91 = aten.convolution_backward(buf89, relu_15, primals_77, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False])
        del primals_77
        buf92 = buf91[0]
        buf93 = buf91[1]
        del buf91
        buf94 = buf85; del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_32.run(relu_15, buf80, buf92, buf94, 192, 1568, grid=grid(192), stream=stream0)
        buf95 = reinterpret_tensor(buf84, (192, 13), (13, 1), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_33.run(relu_15, buf80, buf92, convolution_27, unsqueeze_178, buf95, 2496, 121, grid=grid(2496), stream=stream0)
        buf96 = empty((192, ), device='cuda', dtype=torch.float32)
        buf98 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_34.run(buf95, squeeze_46, buf96, buf98, 192, 13, grid=grid(192), stream=stream0)
        buf97 = reinterpret_tensor(buf89, (8, 192, 14, 14), (37632, 1, 2688, 192), 0); del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_35.run(relu_15, buf80, buf92, convolution_27, unsqueeze_178, buf96, squeeze_46, buf94, primals_31, buf97, 1568, 192, grid=grid(1568, 192), stream=stream0)
        del buf92
        del convolution_27
        del primals_31
        del relu_15
        del squeeze_46
        del unsqueeze_178
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf99 = aten.convolution_backward(buf97, convolution_26, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf97
        del convolution_26
        del primals_76
        buf100 = buf99[0]
        buf101 = buf99[1]
        del buf99
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf102 = aten.convolution_backward(buf100, relu_14, primals_75, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False])
        del primals_75
        buf103 = buf102[0]
        buf104 = buf102[1]
        del buf102
        buf105 = buf96; del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_36.run(relu_14, buf80, buf103, buf105, 192, 1568, grid=grid(192), stream=stream0)
        buf106 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_37.run(relu_14, buf80, buf103, convolution_25, unsqueeze_190, buf106, 2496, 121, grid=grid(2496), stream=stream0)
        buf107 = empty((192, ), device='cuda', dtype=torch.float32)
        buf109 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_34.run(buf106, squeeze_43, buf107, buf109, 192, 13, grid=grid(192), stream=stream0)
        buf108 = reinterpret_tensor(buf100, (8, 192, 14, 14), (37632, 1, 2688, 192), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_38.run(relu_14, buf80, buf103, convolution_25, unsqueeze_190, buf107, squeeze_43, buf105, primals_29, buf108, 1568, 192, grid=grid(1568, 192), stream=stream0)
        del buf103
        del convolution_25
        del primals_29
        del relu_14
        del squeeze_43
        del unsqueeze_190
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf110 = aten.convolution_backward(buf108, convolution_24, primals_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf108
        del convolution_24
        del primals_74
        buf111 = buf110[0]
        buf112 = buf110[1]
        del buf110
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf113 = aten.convolution_backward(buf111, relu_13, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False])
        del primals_73
        buf114 = buf113[0]
        buf115 = buf113[1]
        del buf113
        buf116 = buf106; del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_39.run(relu_13, buf114, buf116, 2496, 121, grid=grid(2496), stream=stream0)
        buf117 = buf107; del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_28.run(buf116, buf117, 192, 13, grid=grid(192), stream=stream0)
        buf118 = reinterpret_tensor(buf116, (192, 13), (1, 192), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_40.run(relu_13, buf114, convolution_23, unsqueeze_202, buf118, 2496, 121, grid=grid(2496), stream=stream0)
        buf119 = empty((192, ), device='cuda', dtype=torch.float32)
        buf120 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_30.run(buf118, squeeze_40, buf119, buf120, 192, 13, grid=grid(192), stream=stream0)
        del buf118
        buf121 = reinterpret_tensor(buf111, (8, 192, 14, 14), (37632, 1, 2688, 192), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_41.run(relu_13, buf114, convolution_23, unsqueeze_202, buf119, squeeze_40, buf117, primals_27, buf121, 1568, 192, grid=grid(1568, 192), stream=stream0)
        del buf114
        del buf119
        del convolution_23
        del primals_27
        del relu_13
        del squeeze_40
        del unsqueeze_202
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf122 = aten.convolution_backward(buf121, getitem_28, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf121
        del getitem_28
        del primals_72
        buf123 = buf122[0]
        buf124 = buf122[1]
        del buf122
        buf125 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_42.run(buf125, buf80, 802816, grid=grid(802816), stream=stream0)
        del buf80
        # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
        buf126 = aten.max_pool2d_with_indices_backward(buf125, mul_92, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_29)
        del buf125
        del getitem_29
        del mul_92
        buf127 = buf126
        del buf126
        buf128 = empty_strided((8, 512, 1, 1, 7), (3584, 1, 28672, 28672, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_43.run(buf127, relu_12, buf128, 28672, 112, grid=grid(28672), stream=stream0)
        buf129 = reinterpret_tensor(buf11, (8, 512, 1, 1), (512, 1, 4096, 4096), 0); del buf11  # reuse
        buf130 = reinterpret_tensor(buf129, (8, 512, 1, 1), (512, 1, 512, 512), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_mul_sum_44.run(buf130, buf128, bitwise_and_2, 4096, 7, grid=grid(4096), stream=stream0)
        del bitwise_and_2
        del buf128
        buf131 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_45.run(buf130, buf131, 512, 8, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf132 = aten.convolution_backward(buf130, mean_1, primals_70, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf130
        del mean_1
        del primals_70
        buf133 = buf132[0]
        buf134 = buf132[1]
        del buf132
        buf135 = empty_strided((512, 49), (1, 512), device='cuda', dtype=torch.float32)
        buf137 = empty_strided((512, 49), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_46.run(relu_12, buf127, div_1, buf133, convolution_21, unsqueeze_214, buf135, buf137, 25088, 128, grid=grid(25088), stream=stream0)
        buf136 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_47.run(buf135, buf136, 512, 49, grid=grid(512), stream=stream0)
        del buf135
        buf138 = empty((512, ), device='cuda', dtype=torch.float32)
        buf140 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_48.run(buf137, squeeze_37, buf138, buf140, 512, 49, grid=grid(512), stream=stream0)
        buf139 = buf127; del buf127  # reuse
        buf141 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_49.run(buf141, relu_12, div_1, buf133, convolution_21, unsqueeze_214, buf138, squeeze_37, buf136, primals_25, 3211264, grid=grid(3211264), stream=stream0)
        del buf133
        del convolution_21
        del div_1
        del primals_25
        del relu_12
        del squeeze_37
        del unsqueeze_214
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf142 = aten.convolution_backward(buf141, cat_1, primals_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_1
        del primals_69
        buf143 = buf142[0]
        buf144 = buf142[1]
        del buf142
        buf145 = empty((160, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_50.run(le_11, buf143, buf145, 7840, 128, grid=grid(7840), stream=stream0)
        buf146 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_51.run(buf145, buf146, 160, 49, grid=grid(160), stream=stream0)
        buf147 = reinterpret_tensor(buf145, (160, 49), (1, 160), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_52.run(le_11, buf143, convolution_20, unsqueeze_226, buf147, 7840, 128, grid=grid(7840), stream=stream0)
        buf148 = empty((160, ), device='cuda', dtype=torch.float32)
        buf149 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_53.run(buf147, squeeze_34, buf148, buf149, 160, 49, grid=grid(160), stream=stream0)
        buf150 = empty_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_54.run(le_11, buf143, convolution_20, unsqueeze_226, buf148, squeeze_34, buf146, primals_23, buf150, 6272, 160, grid=grid(6272, 160), stream=stream0)
        del convolution_20
        del le_11
        del primals_23
        del squeeze_34
        del unsqueeze_226
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf151 = aten.convolution_backward(buf150, convolution_19, primals_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf150
        del convolution_19
        del primals_68
        buf152 = buf151[0]
        buf153 = buf151[1]
        del buf151
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf154 = aten.convolution_backward(buf152, relu_10, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 160, [True, True, False])
        del primals_67
        buf155 = buf154[0]
        buf156 = buf154[1]
        del buf154
        buf157 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_55.run(relu_10, buf143, buf155, buf157, 160, 6272, grid=grid(160), stream=stream0)
        buf158 = reinterpret_tensor(buf147, (160, 49), (49, 1), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_56.run(relu_10, buf143, buf155, convolution_18, unsqueeze_238, buf158, 7840, 128, grid=grid(7840), stream=stream0)
        buf159 = empty((160, ), device='cuda', dtype=torch.float32)
        buf161 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_57.run(buf158, squeeze_31, buf159, buf161, 160, 49, grid=grid(160), stream=stream0)
        buf160 = reinterpret_tensor(buf152, (8, 160, 28, 28), (125440, 1, 4480, 160), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_58.run(relu_10, buf143, buf155, convolution_18, unsqueeze_238, buf159, squeeze_31, buf157, primals_21, buf160, 6272, 160, grid=grid(6272, 160), stream=stream0)
        del buf155
        del convolution_18
        del primals_21
        del relu_10
        del squeeze_31
        del unsqueeze_238
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf162 = aten.convolution_backward(buf160, convolution_17, primals_66, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf160
        del convolution_17
        del primals_66
        buf163 = buf162[0]
        buf164 = buf162[1]
        del buf162
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf165 = aten.convolution_backward(buf163, relu_9, primals_65, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 160, [True, True, False])
        del primals_65
        buf166 = buf165[0]
        buf167 = buf165[1]
        del buf165
        buf168 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_59.run(relu_9, buf143, buf166, buf168, 160, 6272, grid=grid(160), stream=stream0)
        buf169 = buf158; del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_60.run(relu_9, buf143, buf166, convolution_16, unsqueeze_250, buf169, 7840, 128, grid=grid(7840), stream=stream0)
        buf170 = empty((160, ), device='cuda', dtype=torch.float32)
        buf172 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_57.run(buf169, squeeze_28, buf170, buf172, 160, 49, grid=grid(160), stream=stream0)
        buf171 = reinterpret_tensor(buf163, (8, 160, 28, 28), (125440, 1, 4480, 160), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_61.run(relu_9, buf143, buf166, convolution_16, unsqueeze_250, buf170, squeeze_28, buf168, primals_19, buf171, 6272, 160, grid=grid(6272, 160), stream=stream0)
        del buf166
        del convolution_16
        del primals_19
        del relu_9
        del squeeze_28
        del unsqueeze_250
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf173 = aten.convolution_backward(buf171, convolution_15, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf171
        del convolution_15
        del primals_64
        buf174 = buf173[0]
        buf175 = buf173[1]
        del buf173
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf176 = aten.convolution_backward(buf174, relu_8, primals_63, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 160, [True, True, False])
        del primals_63
        buf177 = buf176[0]
        buf178 = buf176[1]
        del buf176
        buf179 = buf169; del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_62.run(relu_8, buf177, buf179, 7840, 128, grid=grid(7840), stream=stream0)
        buf180 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_51.run(buf179, buf180, 160, 49, grid=grid(160), stream=stream0)
        buf181 = reinterpret_tensor(buf179, (160, 49), (1, 160), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_63.run(relu_8, buf177, convolution_14, unsqueeze_262, buf181, 7840, 128, grid=grid(7840), stream=stream0)
        buf182 = empty((160, ), device='cuda', dtype=torch.float32)
        buf183 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_53.run(buf181, squeeze_25, buf182, buf183, 160, 49, grid=grid(160), stream=stream0)
        del buf181
        buf184 = reinterpret_tensor(buf174, (8, 160, 28, 28), (125440, 1, 4480, 160), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_64.run(relu_8, buf177, convolution_14, unsqueeze_262, buf182, squeeze_25, buf180, primals_17, buf184, 6272, 160, grid=grid(6272, 160), stream=stream0)
        del buf177
        del buf182
        del convolution_14
        del primals_17
        del relu_8
        del squeeze_25
        del unsqueeze_262
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf185 = aten.convolution_backward(buf184, getitem_16, primals_62, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf184
        del getitem_16
        del primals_62
        buf186 = buf185[0]
        buf187 = buf185[1]
        del buf185
        buf188 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_65.run(buf188, buf143, 1605632, grid=grid(1605632), stream=stream0)
        del buf143
        # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
        buf189 = aten.max_pool2d_with_indices_backward(buf188, mul_56, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_17)
        del getitem_17
        del mul_56
        buf190 = buf189
        del buf189
        buf191 = empty_strided((8, 256, 1, 1, 25), (6400, 1, 51200, 51200, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_66.run(buf190, relu_7, buf191, 51200, 126, grid=grid(51200), stream=stream0)
        buf192 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf193 = reinterpret_tensor(buf192, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_mul_sum_67.run(buf193, buf191, bitwise_and_3, 2048, 25, grid=grid(2048), stream=stream0)
        del bitwise_and_3
        del buf191
        buf194 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_68.run(buf193, buf194, 256, 8, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf195 = aten.convolution_backward(buf193, mean, primals_60, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf193
        del mean
        del primals_60
        buf196 = buf195[0]
        buf197 = buf195[1]
        del buf195
        buf198 = empty_strided((256, 196), (1, 256), device='cuda', dtype=torch.float32)
        buf200 = empty_strided((256, 196), (1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_69.run(relu_7, buf190, div, buf196, convolution_12, unsqueeze_274, buf198, buf200, 50176, 128, grid=grid(50176), stream=stream0)
        buf199 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_70.run(buf198, buf199, 256, 196, grid=grid(256), stream=stream0)
        del buf198
        buf201 = empty((256, ), device='cuda', dtype=torch.float32)
        buf203 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_71.run(buf200, squeeze_22, buf201, buf203, 256, 196, grid=grid(256), stream=stream0)
        buf202 = buf190; del buf190  # reuse
        buf204 = buf202; del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardsigmoid_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_72.run(buf204, relu_7, div, buf196, convolution_12, unsqueeze_274, buf201, squeeze_22, buf199, primals_15, 6422528, grid=grid(6422528), stream=stream0)
        del buf196
        del convolution_12
        del div
        del primals_15
        del relu_7
        del squeeze_22
        del unsqueeze_274
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf205 = aten.convolution_backward(buf204, cat, primals_59, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat
        del primals_59
        buf206 = buf205[0]
        buf207 = buf205[1]
        del buf205
        buf208 = reinterpret_tensor(buf137, (128, 196), (196, 1), 0); del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_73.run(le_16, buf206, buf208, 25088, 128, grid=grid(25088), stream=stream0)
        buf209 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_74.run(buf208, buf209, 128, 196, grid=grid(128), stream=stream0)
        buf210 = reinterpret_tensor(buf208, (128, 196), (1, 128), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_75.run(le_16, buf206, convolution_11, unsqueeze_286, buf210, 25088, 128, grid=grid(25088), stream=stream0)
        buf211 = empty((128, ), device='cuda', dtype=torch.float32)
        buf212 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_76.run(buf210, squeeze_19, buf211, buf212, 128, 196, grid=grid(128), stream=stream0)
        buf213 = reinterpret_tensor(buf141, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_77.run(le_16, buf206, convolution_11, unsqueeze_286, buf211, squeeze_19, buf209, primals_13, buf213, 25088, 128, grid=grid(25088, 128), stream=stream0)
        del convolution_11
        del le_16
        del primals_13
        del squeeze_19
        del unsqueeze_286
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf214 = aten.convolution_backward(buf213, convolution_10, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf213
        del convolution_10
        del primals_58
        buf215 = buf214[0]
        buf216 = buf214[1]
        del buf214
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf217 = aten.convolution_backward(buf215, relu_5, primals_57, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False])
        del primals_57
        buf218 = buf217[0]
        buf219 = buf217[1]
        del buf217
        buf220 = reinterpret_tensor(buf138, (128, 4), (1, 128), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_78.run(relu_5, buf206, buf218, buf220, 512, 6272, grid=grid(512), stream=stream0)
        buf221 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_79.run(buf220, buf221, 128, 4, grid=grid(128), stream=stream0)
        buf222 = reinterpret_tensor(buf210, (128, 196), (196, 1), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_80.run(relu_5, buf206, buf218, convolution_9, unsqueeze_298, buf222, 25088, 128, grid=grid(25088), stream=stream0)
        buf223 = empty((128, ), device='cuda', dtype=torch.float32)
        buf225 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_81.run(buf222, squeeze_16, buf223, buf225, 128, 196, grid=grid(128), stream=stream0)
        buf224 = reinterpret_tensor(buf215, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_82.run(relu_5, buf206, buf218, convolution_9, unsqueeze_298, buf223, squeeze_16, buf221, primals_11, buf224, 25088, 128, grid=grid(25088, 128), stream=stream0)
        del buf218
        del convolution_9
        del primals_11
        del relu_5
        del squeeze_16
        del unsqueeze_298
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf226 = aten.convolution_backward(buf224, convolution_8, primals_56, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf224
        del convolution_8
        del primals_56
        buf227 = buf226[0]
        buf228 = buf226[1]
        del buf226
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf229 = aten.convolution_backward(buf227, relu_4, primals_55, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False])
        del primals_55
        buf230 = buf229[0]
        buf231 = buf229[1]
        del buf229
        buf232 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_83.run(relu_4, buf206, buf230, buf232, 512, 6272, grid=grid(512), stream=stream0)
        buf233 = buf223; del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_79.run(buf232, buf233, 128, 4, grid=grid(128), stream=stream0)
        del buf232
        buf234 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_84.run(relu_4, buf206, buf230, convolution_7, unsqueeze_310, buf234, 25088, 128, grid=grid(25088), stream=stream0)
        buf235 = empty((128, ), device='cuda', dtype=torch.float32)
        buf237 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_81.run(buf234, squeeze_13, buf235, buf237, 128, 196, grid=grid(128), stream=stream0)
        buf236 = reinterpret_tensor(buf227, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_85.run(relu_4, buf206, buf230, convolution_7, unsqueeze_310, buf235, squeeze_13, buf233, primals_9, buf236, 25088, 128, grid=grid(25088, 128), stream=stream0)
        del buf230
        del convolution_7
        del primals_9
        del relu_4
        del squeeze_13
        del unsqueeze_310
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf238 = aten.convolution_backward(buf236, convolution_6, primals_54, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf236
        del convolution_6
        del primals_54
        buf239 = buf238[0]
        buf240 = buf238[1]
        del buf238
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf241 = aten.convolution_backward(buf239, relu_3, primals_53, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False])
        del primals_53
        buf242 = buf241[0]
        buf243 = buf241[1]
        del buf241
        buf244 = buf234; del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_86.run(relu_3, buf242, buf244, 25088, 128, grid=grid(25088), stream=stream0)
        buf245 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_74.run(buf244, buf245, 128, 196, grid=grid(128), stream=stream0)
        buf246 = reinterpret_tensor(buf244, (128, 196), (1, 128), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_87.run(relu_3, buf242, convolution_5, unsqueeze_322, buf246, 25088, 128, grid=grid(25088), stream=stream0)
        buf247 = empty((128, ), device='cuda', dtype=torch.float32)
        buf248 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_76.run(buf246, squeeze_10, buf247, buf248, 128, 196, grid=grid(128), stream=stream0)
        del buf246
        buf249 = reinterpret_tensor(buf239, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_88.run(relu_3, buf242, convolution_5, unsqueeze_322, buf247, squeeze_10, buf245, primals_7, buf249, 25088, 128, grid=grid(25088, 128), stream=stream0)
        del buf242
        del buf247
        del convolution_5
        del primals_7
        del relu_3
        del squeeze_10
        del unsqueeze_322
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf250 = aten.convolution_backward(buf249, relu_2, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf249
        del primals_52
        buf251 = buf250[0]
        buf252 = buf250[1]
        del buf250
        buf253 = reinterpret_tensor(buf201, (64, 4), (1, 64), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_89.run(relu_2, buf206, buf251, buf253, 256, 6272, grid=grid(256), stream=stream0)
        buf254 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_90.run(buf253, buf254, 64, 4, grid=grid(64), stream=stream0)
        del buf253
        buf255 = empty((64, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_91.run(relu_2, buf206, buf251, convolution_4, unsqueeze_334, buf255, 12544, 128, grid=grid(12544), stream=stream0)
        buf256 = empty((64, ), device='cuda', dtype=torch.float32)
        buf258 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_92.run(buf255, squeeze_7, buf256, buf258, 64, 196, grid=grid(64), stream=stream0)
        del buf255
        buf257 = reinterpret_tensor(buf188, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_93.run(relu_2, buf206, buf251, convolution_4, unsqueeze_334, buf256, squeeze_7, buf254, primals_5, buf257, 25088, 64, grid=grid(25088, 64), stream=stream0)
        del buf206
        del buf251
        del convolution_4
        del primals_5
        del relu_2
        del squeeze_7
        del unsqueeze_334
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf259 = aten.convolution_backward(buf257, convolution_3, primals_51, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf257
        del convolution_3
        del primals_51
        buf260 = buf259[0]
        buf261 = buf259[1]
        del buf259
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf262 = aten.convolution_backward(buf260, relu_1, primals_50, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
        del buf260
        del primals_50
        buf263 = buf262[0]
        buf264 = buf262[1]
        del buf262
        buf265 = reinterpret_tensor(buf200, (64, 784), (784, 1), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_94.run(relu_1, buf263, buf265, 50176, 128, grid=grid(50176), stream=stream0)
        buf266 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_95.run(buf265, buf266, 64, 784, grid=grid(64), stream=stream0)
        buf267 = reinterpret_tensor(buf265, (64, 784), (1, 64), 0); del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_96.run(relu_1, buf263, convolution_2, unsqueeze_346, buf267, 50176, 128, grid=grid(50176), stream=stream0)
        buf268 = empty((64, ), device='cuda', dtype=torch.float32)
        buf269 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_97.run(buf267, squeeze_4, buf268, buf269, 64, 784, grid=grid(64), stream=stream0)
        buf270 = reinterpret_tensor(buf204, (8, 64, 112, 112), (802816, 1, 7168, 64), 0); del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_98.run(relu_1, buf263, convolution_2, unsqueeze_346, buf268, squeeze_4, buf266, primals_3, buf270, 100352, 64, grid=grid(100352, 64), stream=stream0)
        del buf263
        del convolution_2
        del primals_3
        del relu_1
        del squeeze_4
        del unsqueeze_346
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf271 = aten.convolution_backward(buf270, convolution_1, primals_49, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf270
        del convolution_1
        del primals_49
        buf272 = buf271[0]
        buf273 = buf271[1]
        del buf271
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf274 = aten.convolution_backward(buf272, relu, primals_48, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
        del primals_48
        buf275 = buf274[0]
        buf276 = buf274[1]
        del buf274
        buf277 = reinterpret_tensor(buf267, (64, 784), (784, 1), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_94.run(relu, buf275, buf277, 50176, 128, grid=grid(50176), stream=stream0)
        buf278 = buf268; del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_95.run(buf277, buf278, 64, 784, grid=grid(64), stream=stream0)
        buf279 = reinterpret_tensor(buf277, (64, 784), (1, 64), 0); del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_96.run(relu, buf275, convolution, unsqueeze_358, buf279, 50176, 128, grid=grid(50176), stream=stream0)
        buf280 = empty((64, ), device='cuda', dtype=torch.float32)
        buf281 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_97.run(buf279, squeeze_1, buf280, buf281, 64, 784, grid=grid(64), stream=stream0)
        del buf279
        buf282 = reinterpret_tensor(buf272, (8, 64, 112, 112), (802816, 1, 7168, 64), 0); del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_98.run(relu, buf275, convolution, unsqueeze_358, buf280, squeeze_1, buf278, primals_1, buf282, 100352, 64, grid=grid(100352, 64), stream=stream0)
        del buf275
        del buf280
        del convolution
        del primals_1
        del relu
        del squeeze_1
        del unsqueeze_358
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf283 = aten.convolution_backward(buf282, primals_163, primals_47, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf282
        del primals_163
        del primals_47
        buf284 = buf283[1]
        return (buf281, buf278, buf269, buf266, buf258, buf254, buf248, buf245, buf237, buf233, buf225, buf221, buf212, buf209, buf203, buf199, buf183, buf180, buf172, buf168, buf161, buf157, buf149, buf146, buf140, buf136, buf120, buf117, buf109, buf105, buf98, buf94, buf86, buf83, buf77, buf73, buf57, buf54, buf46, buf42, buf35, buf31, buf23, buf20, buf14, buf10, buf284, buf276, buf273, buf264, buf261, buf252, buf243, buf240, buf231, buf228, buf219, buf216, buf207, buf197, buf194, buf187, buf178, buf175, buf167, buf164, buf156, buf153, buf144, buf134, buf131, buf124, buf115, buf112, buf104, buf101, buf93, buf90, buf81, buf71, buf68, buf61, buf52, buf49, buf41, buf38, buf30, buf27, buf18, buf8, buf5, reinterpret_tensor(buf1, (1000, 1024), (1024, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((256, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((160, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((192, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((224, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((1024, 1440, 1, 1), (1440, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((8, 448, 56, 56), (1404928, 1, 25088, 448), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    div = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    mul_56 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    getitem_16 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.int64)
    convolution_14 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((8, 736, 28, 28), (577024, 1, 20608, 736), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    mean_1 = rand_strided((8, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((8, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    mul_92 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    getitem_28 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    getitem_29 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.int64)
    convolution_23 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((8, 1088, 14, 14), (213248, 1, 15232, 1088), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 768, 14, 14), (150528, 1, 10752, 768), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((8, 768, 14, 14), (150528, 1, 10752, 768), device='cuda:0', dtype=torch.float32)
    mean_2 = rand_strided((8, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((8, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    mul_128 = rand_strided((8, 768, 14, 14), (150528, 1, 10752, 768), device='cuda:0', dtype=torch.float32)
    getitem_40 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    getitem_41 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.int64)
    convolution_32 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_3 = rand_strided((8, 1440, 7, 7), (70560, 1, 10080, 1440), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda:0', dtype=torch.float32)
    mean_3 = rand_strided((8, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((8, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((8, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and = rand_strided((8, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bool)
    unsqueeze_94 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_1 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda:0', dtype=torch.bool)
    unsqueeze_106 = rand_strided((1, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_118 = rand_strided((1, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_130 = rand_strided((1, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_142 = rand_strided((1, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_1 = rand_strided((8, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.bool)
    unsqueeze_154 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_6 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_166 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_178 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_190 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_202 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_2 = rand_strided((8, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bool)
    unsqueeze_214 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_11 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cuda:0', dtype=torch.bool)
    unsqueeze_226 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_238 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_3 = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bool)
    unsqueeze_274 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_16 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.bool)
    unsqueeze_286 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_358 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_163, convolution, squeeze_1, relu, convolution_1, convolution_2, squeeze_4, relu_1, convolution_3, convolution_4, squeeze_7, relu_2, convolution_5, squeeze_10, relu_3, convolution_6, convolution_7, squeeze_13, relu_4, convolution_8, convolution_9, squeeze_16, relu_5, convolution_10, convolution_11, squeeze_19, cat, convolution_12, squeeze_22, relu_7, mean, div, mul_56, getitem_16, getitem_17, convolution_14, squeeze_25, relu_8, convolution_15, convolution_16, squeeze_28, relu_9, convolution_17, convolution_18, squeeze_31, relu_10, convolution_19, convolution_20, squeeze_34, cat_1, convolution_21, squeeze_37, relu_12, mean_1, div_1, mul_92, getitem_28, getitem_29, convolution_23, squeeze_40, relu_13, convolution_24, convolution_25, squeeze_43, relu_14, convolution_26, convolution_27, squeeze_46, relu_15, convolution_28, convolution_29, squeeze_49, cat_2, convolution_30, squeeze_52, relu_17, mean_2, div_2, mul_128, getitem_40, getitem_41, convolution_32, squeeze_55, relu_18, convolution_33, convolution_34, squeeze_58, relu_19, convolution_35, convolution_36, squeeze_61, relu_20, convolution_37, convolution_38, squeeze_64, cat_3, convolution_39, squeeze_67, relu_22, mean_3, div_3, clone, permute_1, bitwise_and, unsqueeze_94, le_1, unsqueeze_106, unsqueeze_118, unsqueeze_130, unsqueeze_142, bitwise_and_1, unsqueeze_154, le_6, unsqueeze_166, unsqueeze_178, unsqueeze_190, unsqueeze_202, bitwise_and_2, unsqueeze_214, le_11, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, bitwise_and_3, unsqueeze_274, le_16, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('ese_vovnet19b_dw', benchmark_compiled_module)
