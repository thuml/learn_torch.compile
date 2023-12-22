
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


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xz3sl5pw5l6ayyku3kcca7jkxnnppufjzrllvqrd2js2yaokg3.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward]

triton_poi_fused_hardswish_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp5 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tl.store(in_out_ptr0 + (x0), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d7/cd72vkcuoxrgalxkx3rcgektxdr2d6d7y7pskahmfl2zoylevg4f.py
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
    size_hints=[2048, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1280*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sq/csqnl5aruyo3j5mpd7murjesn6rsy43xzvas2soqukjrdn2i2cbu.py
# Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_div_hardswish_backward_native_batch_norm_backward_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_hardswish_backward_native_batch_norm_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp19 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + (x0 + (256*(r2 // 49)) + (512*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr2 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = 49.0
        tmp7 = tmp5 / tmp6
        tmp8 = tmp0 / tmp3
        tmp9 = 0.5
        tmp10 = tmp8 + tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.where(tmp4, tmp11, tmp7)
        tmp13 = 0.0
        tmp14 = tl.where(tmp2, tmp13, tmp12)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
        tmp20 = tmp18 - tmp19
        tmp21 = tmp14 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4t/c4t6ad3aeb33pk66x4mcqcrmjsxhkdwxu3mf4rghlyis23phy5vr.py
# Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/ch/cch2mvr6rlfrinry3dmhwqx2r2okurg754l6vk6b3qcibsn6ebzd.py
# Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_div_hardswish_backward_native_batch_norm_backward_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_hardswish_backward_native_batch_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ut/cutvknx6e4eazi3lngmjobpnk3at7i2t7x6lwnxgscnnpnzivesl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 256
    x2 = (xindex // 12544)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (x3), None)
    tmp16 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = 49.0
    tmp7 = tmp5 / tmp6
    tmp8 = tmp0 / tmp3
    tmp9 = 0.5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tl.where(tmp4, tmp11, tmp7)
    tmp13 = 0.0
    tmp14 = tl.where(tmp2, tmp13, tmp12)
    tmp17 = tmp15 - tmp16
    tmp19 = 0.002551020408163265
    tmp20 = tmp18 * tmp19
    tmp22 = tmp21 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp17 * tmp23
    tmp25 = tmp14 - tmp24
    tmp27 = tmp26 * tmp19
    tmp28 = tmp25 - tmp27
    tmp30 = tmp21 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x3), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vk/cvknqxmor4umc2ygtiragkcpwts3fuuafjp57zz6xhqpmrgncp22.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_7', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*r2) + (12544*x1)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = 0.16666666666666666
    tmp9 = tmp6 * tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp7, tmp9, tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qv/cqvawzhjyqgeib4sffscj32vhw3zaus5edo7pfecsqppapsmhxqx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_8', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/qq/cqq4lzq7nm7fum4hwkqs4kb4ocmgxkfsyeooxt2tsivdtt75ljxr.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]

triton_poi_fused_hardswish_backward_threshold_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_threshold_backward_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ze/czer2dfxefvgxynhnaogybozmybno55t2v75wkrr374liknlmxxb.py
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
    size_hints=[64, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 8
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/dd/cddqzhh3wgxcn53acpopdbdluxthekjznt2dhqi6pokp3bbfnayf.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp23 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((49*x0) + (12544*(r2 // 49)) + (25088*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (256*(r2 // 49)) + (512*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (256*(r2 // 49)) + (512*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr4 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp7 = tmp5 * tmp6
        tmp9 = 49.0
        tmp10 = tmp8 / tmp9
        tmp11 = tmp7 + tmp10
        tmp12 = tmp0 / tmp3
        tmp13 = 0.5
        tmp14 = tmp12 + tmp13
        tmp15 = tmp11 * tmp14
        tmp16 = tl.where(tmp4, tmp15, tmp11)
        tmp17 = 0.0
        tmp18 = tl.where(tmp2, tmp17, tmp16)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
        tmp24 = tmp22 - tmp23
        tmp25 = tmp18 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp20, xmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ly/cly3abqr4yiktcczc2yt3tiqkygqxfqx76uuvsa4wo26feke2iwv.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (49*x2) + (12544*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (256*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (256*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp7 = tmp5 * tmp6
    tmp9 = 49.0
    tmp10 = tmp8 / tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tmp0 / tmp3
    tmp13 = 0.5
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp4, tmp15, tmp11)
    tmp17 = 0.0
    tmp18 = tl.where(tmp2, tmp17, tmp16)
    tmp21 = tmp19 - tmp20
    tmp23 = 0.002551020408163265
    tmp24 = tmp22 * tmp23
    tmp26 = tmp25 * tmp25
    tmp27 = tmp24 * tmp26
    tmp28 = tmp21 * tmp27
    tmp29 = tmp18 - tmp28
    tmp31 = tmp30 * tmp23
    tmp32 = tmp29 - tmp31
    tmp34 = tmp25 * tmp33
    tmp35 = tmp32 * tmp34
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (256*y3)), tmp35, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dy/cdyevk3ir6754hyibr7fgtjo2wqbgqv43kvwbwuu3oesmw4bxoi7.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((49*x0) + (12544*(r2 // 49)) + (25088*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr2 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp12 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q5/cq5uiypbtbuesrtneesusztxgmpgaerpe6e4e2iiu4bw7nx66eku.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (49*x2) + (12544*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp15 = tmp13 - tmp14
    tmp17 = 0.002551020408163265
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp12 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp19 * tmp27
    tmp29 = tmp26 * tmp28
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xm/cxm4c76tzuiywkvn4kgakyvng4n673wiqszjicq2qv6xkjphmg6j.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*r2) + (6272*x1)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = 0.16666666666666666
    tmp9 = tmp6 * tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp7, tmp9, tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q3/cq3ttv7wkuszloot3krsan6kfzle2w74azmdm4vxzusww7gwl273.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/gl/cgl7dpmg354t2oeja265eqzrhmusw3d43mrkfrcj6zmvriyyd5ye.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]

triton_poi_fused_hardswish_backward_threshold_backward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_threshold_backward_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fr/cfru3v73324s7j5p7swhp332ts3kywpjtqeawz2rzhsqrvebujck.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mk/cmkil3zu2owz6v7xinqw5mnt7iao6vbdwgi5wrthelqek7scceuc.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp23 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (12544*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((49*x0) + (6272*(r2 // 49)) + (12544*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (128*(r2 // 49)) + (256*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (128*(r2 // 49)) + (256*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr4 + (x0 + (128*r2) + (12544*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp7 = tmp5 * tmp6
        tmp9 = 49.0
        tmp10 = tmp8 / tmp9
        tmp11 = tmp7 + tmp10
        tmp12 = tmp0 / tmp3
        tmp13 = 0.5
        tmp14 = tmp12 + tmp13
        tmp15 = tmp11 * tmp14
        tmp16 = tl.where(tmp4, tmp15, tmp11)
        tmp17 = 0.0
        tmp18 = tl.where(tmp2, tmp17, tmp16)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
        tmp24 = tmp22 - tmp23
        tmp25 = tmp18 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp20, xmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ir/cirfpzihkw2fggi3rc3jopqtri5twfif75ya7t6emkrbrhgoomkw.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_20', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/te/cte4xjouyb2p4zusso6bpwhn5qhxneetxu3nkimyvm4msuqs5drp.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zy/czyzcg7kdp4xcmrhbjei5nwqy2hszwr4ufyfzpkssjwdhrqfnnyy.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (49*x2) + (6272*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (128*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (128*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp7 = tmp5 * tmp6
    tmp9 = 49.0
    tmp10 = tmp8 / tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tmp0 / tmp3
    tmp13 = 0.5
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp4, tmp15, tmp11)
    tmp17 = 0.0
    tmp18 = tl.where(tmp2, tmp17, tmp16)
    tmp21 = tmp19 - tmp20
    tmp23 = 0.002551020408163265
    tmp24 = tmp22 * tmp23
    tmp26 = tmp25 * tmp25
    tmp27 = tmp24 * tmp26
    tmp28 = tmp21 * tmp27
    tmp29 = tmp18 - tmp28
    tmp31 = tmp30 * tmp23
    tmp32 = tmp29 - tmp31
    tmp34 = tmp25 * tmp33
    tmp35 = tmp32 * tmp34
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (128*y3)), tmp35, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yy/cyyqzldnqwnm4pl2ttfqp2ceyic5lcybr4qlh7xb6w7tx2metecb.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1664
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (128*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x1) + (25088*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp3 / tmp6
        tmp10 = 0.5
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.where(tmp7, tmp12, tmp8)
        tmp14 = 0.0
        tmp15 = tl.where(tmp5, tmp14, tmp13)
        tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp17 = tl.where(tmp2, tmp15, tmp16)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q3/cq35nrnjkxsc7tulgruhjrjhu33nkgcyc4g2am7al236cmngsfev.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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


# kernel path: /tmp/torchinductor_youkaichao/zz/czzjc7ztyqrgfr2yghdanaiusio5w6rvo44ocd5jomtv5pz663q2.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1664
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x0) + (25088*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp3 / tmp6
        tmp10 = 0.5
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.where(tmp7, tmp12, tmp8)
        tmp14 = 0.0
        tmp15 = tl.where(tmp5, tmp14, tmp13)
        tmp16 = tl.load(in_ptr2 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp15 * tmp18
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxd35woyzozgazbjil7psycesunr7k376hcsmbxic7orvf5ch3y.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r6/cr64vkanac2jhxkt7aharagb3ax3ndczgdjgmamroylggbwcvaqn.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp15 = tmp13 - tmp14
    tmp17 = 0.0006377551020408163
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp12 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp19 * tmp27
    tmp29 = tmp26 * tmp28
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s5/cs5ev4twyn2zkslnezbfnjs6ysxgjuxbb3wablx6fj53lxbcbdvd.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (64*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x1) + (12544*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp3 / tmp6
        tmp10 = 0.5
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.where(tmp7, tmp12, tmp8)
        tmp14 = 0.0
        tmp15 = tl.where(tmp5, tmp14, tmp13)
        tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp17 = tl.where(tmp2, tmp15, tmp16)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ya/cyaula5mixpjbdo73b5h7nnokwtw5d7x3tvayg24zgkbz3qmbynn.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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


# kernel path: /tmp/torchinductor_youkaichao/e2/ce27u6h6x2k3a2blmtzjoxi4m4xvn6kjjrjgejt3mfqqby4cb3qq.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x0) + (12544*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp3 / tmp6
        tmp10 = 0.5
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.where(tmp7, tmp12, tmp8)
        tmp14 = 0.0
        tmp15 = tl.where(tmp5, tmp14, tmp13)
        tmp16 = tl.load(in_ptr2 + (x0 + (64*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp15 * tmp18
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c5/cc5d3ti3y6cwwvhwpb3xrkmx47rdw6rss34mv7h7y2kbj53j62lw.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dv/cdv6ra2jntxqow2g644jg27s4jyk7ojkd56h6szmhthv7iygkwl2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (12544*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp15 = tmp13 - tmp14
    tmp17 = 0.0006377551020408163
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp12 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp19 * tmp27
    tmp29 = tmp26 * tmp28
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jv/cjvmkfllbhfrpk445fwptlryb6l27f6jsveenyfpjgx34e7ys3nr.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((784*x1) + (50176*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jl/cjlmxd5s56sxytmg5yut4spkyvoao5jss6hjdmue3xf6o545ky22.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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


# kernel path: /tmp/torchinductor_youkaichao/6d/c6df3mac4ijcqa57py3lvb364y5o64vgjsuas4dg7tqqlk2rfnmg.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((784*x0) + (50176*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp12 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/st/cst46jfih4wqj36mswo7zj3atluqj7yutup34mh7zo7eubktt3p4.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sp/cspkwpoxnh2nyrdrjj65rluyxgwt2iwzfcytsbrbo6upgfjtmk77.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (784*x2) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp15 = tmp13 - tmp14
    tmp17 = 0.00015943877551020407
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp12 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp19 * tmp27
    tmp29 = tmp26 * tmp28
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jf/cjfci4uftqg5ijjjmtktbff4r4flyg7ueubxg4xr3ojxbwr3ayrp.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (32*r2) + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((784*x1) + (25088*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sc/cscenyagvbxx6iqxhpao3yvnbc5uvu6ijl5rovrhkd55xm5amrbj.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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


# kernel path: /tmp/torchinductor_youkaichao/cn/ccnl2rnub2imaqz4f4czhqecobbr2jc6y7yl7ycejidq47h2w2g6.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((784*x0) + (25088*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp12 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5r/c5ribkpu42c2yjvuihgoczwcsy5kxgpkbqceirz4nl2pyc2net6w.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/md/cmd4cjkbitqupuqizr46csmr3q3mzy6gmxi3pkutiouj4km4lzgj.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 32
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
    tmp0 = tl.load(in_ptr0 + (x2 + (32*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (784*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (32*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp15 = tmp13 - tmp14
    tmp17 = 0.00015943877551020407
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp12 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp19 * tmp27
    tmp29 = tmp26 * tmp28
    tl.store(out_ptr0 + (x2 + (32*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fb/cfbqdkmfzcdli4ilewvshqazjdxcqrjihc7xqfhnm2kbxlqawk62.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (32*r2) + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((3136*x1) + (100352*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6w/c6wuateaqiytghr7yv7hcynbwckru7yfloicqwhhyrps4vmht6qe.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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


# kernel path: /tmp/torchinductor_youkaichao/42/c42zh667hzecotp2yeqplopw5l6f4trgbky2escojxfes4vm7e4r.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((3136*x0) + (100352*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp12 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x3/cx33fj4wrguq52bfrtaob3fipxidswo2ax5bnaxlpkp5n2kk3kh2.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmtomtmwjunszlwxvzie2nblbsvffnljmhyw7ulpj27v7eyqhzp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 32
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
    tmp0 = tl.load(in_ptr0 + (x2 + (32*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (3136*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (32*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp15 = tmp13 - tmp14
    tmp17 = 3.985969387755102e-05
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp12 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp19 * tmp27
    tmp29 = tmp26 * tmp28
    tl.store(out_ptr0 + (x2 + (32*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eb/cebozff3w7ztoxqxzuxfnkaycktpwse3jsnsbjz6fk34ezotswld.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (16*r2) + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((3136*x1) + (50176*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dd/cddjj72foomgscxshcwmisfsopwkufo7nioundwsvgpx5gfreiej.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# kernel path: /tmp/torchinductor_youkaichao/vx/cvxbrwwbcokco2pke5c2xte6po3b4zj7dkwkr4dsnsp7nduiaqd4.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r2) + (2048*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((3136*x0) + (50176*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (16*r2) + (2048*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp12 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f6/cf6vfizidzo62f6evtchduooheifujack4kvzzbptg37siv4v2om.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
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
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/io/cionqyylyyf4rgvj7uhtvna3hu2qddc5sn36vtqthld2fwzsffmq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (3136*x2) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (16*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp15 = tmp13 - tmp14
    tmp17 = 3.985969387755102e-05
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp12 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp19 * tmp27
    tmp29 = tmp26 * tmp28
    tl.store(out_ptr0 + (x2 + (16*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2d/c2d6ewrrjxnszij2fons6ixn3jznmo5vh22ddoohqd2zob4lulkd.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (16*r2) + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((12544*x1) + (200704*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wv/cwvd5w5vme3jw5zbyl7zgjmu6wagwhl6i4cm2kru5thrqlseoiit.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 16
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


# kernel path: /tmp/torchinductor_youkaichao/7d/c7dnwas2kl7okfkzixzwc7tefiagwblvjr2lpdawrdej63apnesx.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r2) + (2048*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((12544*x0) + (200704*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (16*r2) + (2048*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp12 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4d/c4ddrojlvx7n2gp6ihqtfnauyv5lrw77vdfvviulfelsvw33gtfn.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
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
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5t/c5tbpdnlju6cdbbgdoogth52wgsdgharuvw5lhrccxttjbbdy74k.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100352
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (12544*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (16*y3)), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp15 = tmp13 - tmp14
    tmp17 = 9.964923469387754e-06
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp12 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp19 * tmp27
    tmp29 = tmp26 * tmp28
    tl.store(out_ptr0 + (x2 + (16*y3)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yp/cyptofbngryotxa7sgjbwhmsec5srkzte3re2w4ervqedel5y6gc.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (8*r2) + (1024*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((12544*x1) + (100352*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/og/cogzv2vzkoixr4pvjkjsh752by4wh47jsysuihfx6gjq5gynwhqq.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 8
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


# kernel path: /tmp/torchinductor_youkaichao/un/cun7esd35akkmzxy4acyczo6cvwkp4e2gkrjsbgeq3gotsrdrpia.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 8
    x1 = (xindex // 8)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (8*r2) + (1024*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((12544*x0) + (100352*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (8*r2) + (1024*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp12 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vs/cvst4kt5censteuzw655o5tlztqm4ithn76fygns6jomzh5yvhsd.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8
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
        tmp0 = tl.load(in_ptr0 + (x0 + (8*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ig/cig5bykq3uosliqij3nuu2e2s7x353u757rif6swwqd3i6tqh6t2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 8], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100352
    xnumel = 8
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
    tmp0 = tl.load(in_ptr0 + (x2 + (8*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (12544*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (8*y3)), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp15 = tmp13 - tmp14
    tmp17 = 9.964923469387754e-06
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp12 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp19 * tmp27
    tmp29 = tmp26 * tmp28
    tl.store(out_ptr0 + (x2 + (8*y3)), tmp29, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_83, primals_85, primals_86, primals_87, primals_89, primals_91, primals_92, primals_175, convolution, squeeze_1, clone, div, convolution_1, squeeze_4, clone_1, div_1, convolution_2, squeeze_7, clone_2, div_2, convolution_3, squeeze_10, clone_3, div_3, convolution_4, squeeze_13, clone_4, div_4, convolution_5, squeeze_16, clone_5, div_5, convolution_6, squeeze_19, clone_6, div_6, convolution_7, squeeze_22, clone_7, div_7, convolution_8, squeeze_25, clone_8, div_8, convolution_9, squeeze_28, clone_9, div_9, convolution_10, squeeze_31, clone_10, div_10, convolution_11, squeeze_34, clone_11, div_11, convolution_12, squeeze_37, clone_12, div_12, convolution_13, squeeze_40, clone_13, div_13, convolution_14, squeeze_43, clone_14, div_14, convolution_15, squeeze_46, clone_15, div_15, convolution_16, squeeze_49, clone_16, div_16, convolution_17, squeeze_52, clone_17, div_17, convolution_18, squeeze_55, clone_18, div_18, convolution_19, squeeze_58, clone_19, div_19, convolution_20, squeeze_61, clone_20, div_20, convolution_21, squeeze_64, clone_21, div_21, convolution_22, squeeze_67, clone_22, div_22, convolution_23, squeeze_70, clone_23, div_23, mean, relu, div_24, mul_192, convolution_26, squeeze_73, clone_24, div_25, convolution_27, squeeze_76, clone_25, div_26, mean_1, relu_1, div_27, mul_209, convolution_30, squeeze_79, clone_26, mean_2, convolution_31, view_1, permute_1, unsqueeze_110, bitwise_and, unsqueeze_122, unsqueeze_134, bitwise_and_1, unsqueeze_146, unsqueeze_158, unsqueeze_170, unsqueeze_182, unsqueeze_194, unsqueeze_206, unsqueeze_218, unsqueeze_230, unsqueeze_242, unsqueeze_254, unsqueeze_266, unsqueeze_278, unsqueeze_290, unsqueeze_302, unsqueeze_314, unsqueeze_326, unsqueeze_338, unsqueeze_350, unsqueeze_362, unsqueeze_374, unsqueeze_386, unsqueeze_398, unsqueeze_410, unsqueeze_422, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (8, ), (1, ))
    assert_size_stride(primals_3, (8, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_13, (32, ), (1, ))
    assert_size_stride(primals_15, (32, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_27, (128, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_49, (256, ), (1, ))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_57, (8, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_58, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_59, (16, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_60, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_61, (32, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_62, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_63, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_64, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_65, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_66, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_67, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_68, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_69, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_70, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_71, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_72, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_73, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_74, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_75, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_76, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_77, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_78, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_79, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_80, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_81, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_83, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_85, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_86, (256, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_87, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_89, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_91, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_92, (1280, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_175, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(squeeze_1, (8, ), (1, ))
    assert_size_stride(clone, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(div, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(convolution_1, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(squeeze_4, (8, ), (1, ))
    assert_size_stride(clone_1, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(div_1, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(convolution_2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_7, (16, ), (1, ))
    assert_size_stride(clone_2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(div_2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_3, (8, 16, 56, 56), (50176, 1, 896, 16))
    assert_size_stride(squeeze_10, (16, ), (1, ))
    assert_size_stride(clone_3, (8, 16, 56, 56), (50176, 1, 896, 16))
    assert_size_stride(div_3, (8, 16, 56, 56), (50176, 1, 896, 16))
    assert_size_stride(convolution_4, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(squeeze_13, (32, ), (1, ))
    assert_size_stride(clone_4, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(div_4, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(convolution_5, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(squeeze_16, (32, ), (1, ))
    assert_size_stride(clone_5, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(div_5, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(convolution_6, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(squeeze_19, (32, ), (1, ))
    assert_size_stride(clone_6, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(div_6, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(convolution_7, (8, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(squeeze_22, (32, ), (1, ))
    assert_size_stride(clone_7, (8, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(div_7, (8, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(convolution_8, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_25, (64, ), (1, ))
    assert_size_stride(clone_8, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(div_8, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(convolution_9, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_28, (64, ), (1, ))
    assert_size_stride(clone_9, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(div_9, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(convolution_10, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_31, (64, ), (1, ))
    assert_size_stride(clone_10, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(div_10, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(convolution_11, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(squeeze_34, (64, ), (1, ))
    assert_size_stride(clone_11, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(div_11, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(convolution_12, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_37, (128, ), (1, ))
    assert_size_stride(clone_12, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_12, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_13, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_40, (128, ), (1, ))
    assert_size_stride(clone_13, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_13, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_14, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_43, (128, ), (1, ))
    assert_size_stride(clone_14, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_14, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_15, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_46, (128, ), (1, ))
    assert_size_stride(clone_15, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_15, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_16, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_49, (128, ), (1, ))
    assert_size_stride(clone_16, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_16, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_17, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_52, (128, ), (1, ))
    assert_size_stride(clone_17, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_17, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_18, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_55, (128, ), (1, ))
    assert_size_stride(clone_18, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_18, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_19, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_58, (128, ), (1, ))
    assert_size_stride(clone_19, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_19, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_20, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_61, (128, ), (1, ))
    assert_size_stride(clone_20, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_20, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_21, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_64, (128, ), (1, ))
    assert_size_stride(clone_21, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_21, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_22, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_67, (128, ), (1, ))
    assert_size_stride(clone_22, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_22, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_23, (8, 128, 7, 7), (6272, 1, 896, 128))
    assert_size_stride(squeeze_70, (128, ), (1, ))
    assert_size_stride(clone_23, (8, 128, 7, 7), (6272, 1, 896, 128))
    assert_size_stride(div_23, (8, 128, 7, 7), (6272, 1, 896, 128))
    assert_size_stride(mean, (8, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(relu, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_24, (8, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(mul_192, (8, 128, 7, 7), (6272, 1, 896, 128))
    assert_size_stride(convolution_26, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(squeeze_73, (256, ), (1, ))
    assert_size_stride(clone_24, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(div_25, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(convolution_27, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(squeeze_76, (256, ), (1, ))
    assert_size_stride(clone_25, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(div_26, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(mean_1, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(relu_1, (8, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(div_27, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(mul_209, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(convolution_30, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(squeeze_79, (256, ), (1, ))
    assert_size_stride(clone_26, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(mean_2, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_31, (8, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(view_1, (8, 1280), (1280, 1))
    assert_size_stride(permute_1, (1000, 1280), (1280, 1))
    assert_size_stride(unsqueeze_110, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(bitwise_and, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(unsqueeze_122, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_134, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(bitwise_and_1, (8, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(unsqueeze_146, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_158, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_170, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_182, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_194, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_206, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_218, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_230, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_242, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_254, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_266, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_278, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_290, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_302, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_314, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_326, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_338, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_350, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_362, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_374, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_386, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_398, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_410, (1, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(unsqueeze_422, (1, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), view_1, out=buf1)
        del view_1
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = reinterpret_tensor(buf0, (8, 1280, 1, 1), (1280, 1, 1, 1), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_1.run(buf3, convolution_31, 10240, grid=grid(10240), stream=stream0)
        del convolution_31
        buf4 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_2.run(buf3, buf4, 1280, 8, grid=grid(1280), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf5 = aten.convolution_backward(buf3, mean_2, primals_92, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf3
        del mean_2
        del primals_92
        buf6 = buf5[0]
        buf7 = buf5[1]
        del buf5
        buf8 = empty_strided((256, 4), (1, 256), device='cuda', dtype=torch.float32)
        buf10 = empty_strided((256, 4), (1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_div_hardswish_backward_native_batch_norm_backward_3.run(clone_26, buf6, convolution_30, unsqueeze_110, buf8, buf10, 1024, 98, grid=grid(1024), stream=stream0)
        buf9 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4.run(buf8, buf9, 256, 4, grid=grid(256), stream=stream0)
        buf11 = empty((256, ), device='cuda', dtype=torch.float32)
        buf12 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_5.run(buf10, squeeze_79, buf11, buf12, 256, 4, grid=grid(256), stream=stream0)
        buf13 = empty_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_6.run(clone_26, buf6, convolution_30, unsqueeze_110, buf11, squeeze_79, buf9, primals_53, buf13, 100352, grid=grid(100352), stream=stream0)
        del clone_26
        del convolution_30
        del primals_53
        del squeeze_79
        del unsqueeze_110
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf14 = aten.convolution_backward(buf13, mul_209, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_209
        del primals_91
        buf15 = buf14[0]
        buf16 = buf14[1]
        del buf14
        buf17 = reinterpret_tensor(buf6, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf6  # reuse
        buf18 = reinterpret_tensor(buf17, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_7.run(buf18, buf15, div_26, bitwise_and, 2048, 49, grid=grid(2048), stream=stream0)
        del bitwise_and
        del div_26
        buf19 = buf11; del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_8.run(buf18, buf19, 256, 8, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf20 = aten.convolution_backward(buf18, relu_1, primals_89, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf18
        del primals_89
        buf21 = buf20[0]
        buf22 = buf20[1]
        del buf20
        buf23 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]
        triton_poi_fused_hardswish_backward_threshold_backward_9.run(buf23, relu_1, 512, grid=grid(512), stream=stream0)
        del relu_1
        buf24 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf23, buf24, 64, 8, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf25 = aten.convolution_backward(buf23, mean_1, primals_87, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_1
        del primals_87
        buf26 = buf25[0]
        buf27 = buf25[1]
        del buf25
        buf28 = buf10; del buf10  # reuse
        buf30 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_11.run(clone_25, buf15, div_27, buf26, convolution_27, unsqueeze_122, buf28, buf30, 1024, 98, grid=grid(1024), stream=stream0)
        buf29 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4.run(buf28, buf29, 256, 4, grid=grid(256), stream=stream0)
        buf31 = empty((256, ), device='cuda', dtype=torch.float32)
        buf33 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_5.run(buf30, squeeze_76, buf31, buf33, 256, 4, grid=grid(256), stream=stream0)
        buf32 = buf13; del buf13  # reuse
        buf34 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_12.run(buf34, clone_25, buf15, div_27, buf26, convolution_27, unsqueeze_122, buf31, squeeze_76, buf29, primals_51, 392, 256, grid=grid(392, 256), stream=stream0)
        del buf15
        del buf26
        del clone_25
        del convolution_27
        del div_27
        del primals_51
        del squeeze_76
        del unsqueeze_122
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf35 = aten.convolution_backward(buf34, div_25, primals_86, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 256, [True, True, False])
        del div_25
        del primals_86
        buf36 = buf35[0]
        buf37 = buf35[1]
        del buf35
        buf38 = buf30; del buf30  # reuse
        buf40 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_13.run(clone_24, buf36, convolution_26, unsqueeze_134, buf38, buf40, 1024, 98, grid=grid(1024), stream=stream0)
        buf39 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4.run(buf38, buf39, 256, 4, grid=grid(256), stream=stream0)
        del buf38
        buf41 = empty((256, ), device='cuda', dtype=torch.float32)
        buf42 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_5.run(buf40, squeeze_73, buf41, buf42, 256, 4, grid=grid(256), stream=stream0)
        buf43 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_14.run(clone_24, buf36, convolution_26, unsqueeze_134, buf41, squeeze_73, buf39, primals_49, buf43, 392, 256, grid=grid(392, 256), stream=stream0)
        del buf36
        del buf41
        del clone_24
        del convolution_26
        del primals_49
        del squeeze_73
        del unsqueeze_134
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf44 = aten.convolution_backward(buf43, mul_192, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_192
        del primals_85
        buf45 = buf44[0]
        buf46 = buf44[1]
        del buf44
        buf47 = reinterpret_tensor(buf40, (8, 128, 1, 1), (128, 1, 1024, 1024), 0); del buf40  # reuse
        buf48 = reinterpret_tensor(buf47, (8, 128, 1, 1), (128, 1, 128, 128), 0); del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_15.run(buf48, buf45, div_23, bitwise_and_1, 1024, 49, grid=grid(1024), stream=stream0)
        del bitwise_and_1
        del div_23
        buf49 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_16.run(buf48, buf49, 128, 8, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf50 = aten.convolution_backward(buf48, relu, primals_83, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf48
        del primals_83
        buf51 = buf50[0]
        buf52 = buf50[1]
        del buf50
        buf53 = buf51; del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]
        triton_poi_fused_hardswish_backward_threshold_backward_17.run(buf53, relu, 256, grid=grid(256), stream=stream0)
        del relu
        buf54 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_18.run(buf53, buf54, 32, 8, grid=grid(32), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf55 = aten.convolution_backward(buf53, mean, primals_81, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf53
        del mean
        del primals_81
        buf56 = buf55[0]
        buf57 = buf55[1]
        del buf55
        buf58 = reinterpret_tensor(buf23, (128, 4), (1, 128), 0); del buf23  # reuse
        buf60 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_19.run(clone_23, buf45, div_24, buf56, convolution_23, unsqueeze_146, buf58, buf60, 512, 98, grid=grid(512), stream=stream0)
        buf59 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_20.run(buf58, buf59, 128, 4, grid=grid(128), stream=stream0)
        del buf58
        buf61 = empty((128, ), device='cuda', dtype=torch.float32)
        buf63 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_21.run(buf60, squeeze_70, buf61, buf63, 128, 4, grid=grid(128), stream=stream0)
        del buf60
        buf62 = empty_strided((8, 128, 7, 7), (6272, 1, 896, 128), device='cuda', dtype=torch.float32)
        buf64 = buf62; del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_22.run(buf64, clone_23, buf45, div_24, buf56, convolution_23, unsqueeze_146, buf61, squeeze_70, buf59, primals_47, 392, 128, grid=grid(392, 128), stream=stream0)
        del buf45
        del buf56
        del clone_23
        del convolution_23
        del div_24
        del primals_47
        del squeeze_70
        del unsqueeze_146
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf65 = aten.convolution_backward(buf64, div_22, primals_80, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False])
        del buf64
        del div_22
        del primals_80
        buf66 = buf65[0]
        buf67 = buf65[1]
        del buf65
        buf68 = empty((128, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_23.run(clone_22, buf66, buf68, 1664, 121, grid=grid(1664), stream=stream0)
        buf69 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_24.run(buf68, buf69, 128, 13, grid=grid(128), stream=stream0)
        buf70 = reinterpret_tensor(buf68, (128, 13), (1, 128), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_25.run(clone_22, buf66, convolution_22, unsqueeze_158, buf70, 1664, 121, grid=grid(1664), stream=stream0)
        buf71 = empty((128, ), device='cuda', dtype=torch.float32)
        buf72 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_26.run(buf70, squeeze_67, buf71, buf72, 128, 13, grid=grid(128), stream=stream0)
        buf73 = empty_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_27.run(clone_22, buf66, convolution_22, unsqueeze_158, buf71, squeeze_67, buf69, primals_45, buf73, 1568, 128, grid=grid(1568, 128), stream=stream0)
        del buf66
        del clone_22
        del convolution_22
        del primals_45
        del squeeze_67
        del unsqueeze_158
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf74 = aten.convolution_backward(buf73, div_21, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_21
        del primals_79
        buf75 = buf74[0]
        buf76 = buf74[1]
        del buf74
        buf77 = reinterpret_tensor(buf70, (128, 13), (13, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_23.run(clone_21, buf75, buf77, 1664, 121, grid=grid(1664), stream=stream0)
        buf78 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_24.run(buf77, buf78, 128, 13, grid=grid(128), stream=stream0)
        buf79 = reinterpret_tensor(buf77, (128, 13), (1, 128), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_25.run(clone_21, buf75, convolution_21, unsqueeze_170, buf79, 1664, 121, grid=grid(1664), stream=stream0)
        buf80 = empty((128, ), device='cuda', dtype=torch.float32)
        buf81 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_26.run(buf79, squeeze_64, buf80, buf81, 128, 13, grid=grid(128), stream=stream0)
        buf82 = buf73; del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_27.run(clone_21, buf75, convolution_21, unsqueeze_170, buf80, squeeze_64, buf78, primals_43, buf82, 1568, 128, grid=grid(1568, 128), stream=stream0)
        del buf75
        del clone_21
        del convolution_21
        del primals_43
        del squeeze_64
        del unsqueeze_170
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf83 = aten.convolution_backward(buf82, div_20, primals_78, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False])
        del div_20
        del primals_78
        buf84 = buf83[0]
        buf85 = buf83[1]
        del buf83
        buf86 = reinterpret_tensor(buf79, (128, 13), (13, 1), 0); del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_23.run(clone_20, buf84, buf86, 1664, 121, grid=grid(1664), stream=stream0)
        buf87 = buf80; del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_24.run(buf86, buf87, 128, 13, grid=grid(128), stream=stream0)
        buf88 = reinterpret_tensor(buf86, (128, 13), (1, 128), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_25.run(clone_20, buf84, convolution_20, unsqueeze_182, buf88, 1664, 121, grid=grid(1664), stream=stream0)
        buf89 = empty((128, ), device='cuda', dtype=torch.float32)
        buf90 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_26.run(buf88, squeeze_61, buf89, buf90, 128, 13, grid=grid(128), stream=stream0)
        buf91 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_27.run(clone_20, buf84, convolution_20, unsqueeze_182, buf89, squeeze_61, buf87, primals_41, buf91, 1568, 128, grid=grid(1568, 128), stream=stream0)
        del buf84
        del clone_20
        del convolution_20
        del primals_41
        del squeeze_61
        del unsqueeze_182
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf92 = aten.convolution_backward(buf91, div_19, primals_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_19
        del primals_77
        buf93 = buf92[0]
        buf94 = buf92[1]
        del buf92
        buf95 = reinterpret_tensor(buf88, (128, 13), (13, 1), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_23.run(clone_19, buf93, buf95, 1664, 121, grid=grid(1664), stream=stream0)
        buf96 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_24.run(buf95, buf96, 128, 13, grid=grid(128), stream=stream0)
        buf97 = reinterpret_tensor(buf95, (128, 13), (1, 128), 0); del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_25.run(clone_19, buf93, convolution_19, unsqueeze_194, buf97, 1664, 121, grid=grid(1664), stream=stream0)
        buf98 = empty((128, ), device='cuda', dtype=torch.float32)
        buf99 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_26.run(buf97, squeeze_58, buf98, buf99, 128, 13, grid=grid(128), stream=stream0)
        buf100 = buf91; del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_27.run(clone_19, buf93, convolution_19, unsqueeze_194, buf98, squeeze_58, buf96, primals_39, buf100, 1568, 128, grid=grid(1568, 128), stream=stream0)
        del buf93
        del clone_19
        del convolution_19
        del primals_39
        del squeeze_58
        del unsqueeze_194
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf101 = aten.convolution_backward(buf100, div_18, primals_76, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False])
        del div_18
        del primals_76
        buf102 = buf101[0]
        buf103 = buf101[1]
        del buf101
        buf104 = reinterpret_tensor(buf97, (128, 13), (13, 1), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_23.run(clone_18, buf102, buf104, 1664, 121, grid=grid(1664), stream=stream0)
        buf105 = buf98; del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_24.run(buf104, buf105, 128, 13, grid=grid(128), stream=stream0)
        buf106 = reinterpret_tensor(buf104, (128, 13), (1, 128), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_25.run(clone_18, buf102, convolution_18, unsqueeze_206, buf106, 1664, 121, grid=grid(1664), stream=stream0)
        buf107 = empty((128, ), device='cuda', dtype=torch.float32)
        buf108 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_26.run(buf106, squeeze_55, buf107, buf108, 128, 13, grid=grid(128), stream=stream0)
        buf109 = buf100; del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_27.run(clone_18, buf102, convolution_18, unsqueeze_206, buf107, squeeze_55, buf105, primals_37, buf109, 1568, 128, grid=grid(1568, 128), stream=stream0)
        del buf102
        del clone_18
        del convolution_18
        del primals_37
        del squeeze_55
        del unsqueeze_206
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf110 = aten.convolution_backward(buf109, div_17, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_17
        del primals_75
        buf111 = buf110[0]
        buf112 = buf110[1]
        del buf110
        buf113 = reinterpret_tensor(buf106, (128, 13), (13, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_23.run(clone_17, buf111, buf113, 1664, 121, grid=grid(1664), stream=stream0)
        buf114 = buf107; del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_24.run(buf113, buf114, 128, 13, grid=grid(128), stream=stream0)
        buf115 = reinterpret_tensor(buf113, (128, 13), (1, 128), 0); del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_25.run(clone_17, buf111, convolution_17, unsqueeze_218, buf115, 1664, 121, grid=grid(1664), stream=stream0)
        buf116 = empty((128, ), device='cuda', dtype=torch.float32)
        buf117 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_26.run(buf115, squeeze_52, buf116, buf117, 128, 13, grid=grid(128), stream=stream0)
        buf118 = buf109; del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_27.run(clone_17, buf111, convolution_17, unsqueeze_218, buf116, squeeze_52, buf114, primals_35, buf118, 1568, 128, grid=grid(1568, 128), stream=stream0)
        del buf111
        del clone_17
        del convolution_17
        del primals_35
        del squeeze_52
        del unsqueeze_218
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf119 = aten.convolution_backward(buf118, div_16, primals_74, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False])
        del div_16
        del primals_74
        buf120 = buf119[0]
        buf121 = buf119[1]
        del buf119
        buf122 = reinterpret_tensor(buf115, (128, 13), (13, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_23.run(clone_16, buf120, buf122, 1664, 121, grid=grid(1664), stream=stream0)
        buf123 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_24.run(buf122, buf123, 128, 13, grid=grid(128), stream=stream0)
        buf124 = reinterpret_tensor(buf122, (128, 13), (1, 128), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_25.run(clone_16, buf120, convolution_16, unsqueeze_230, buf124, 1664, 121, grid=grid(1664), stream=stream0)
        buf125 = empty((128, ), device='cuda', dtype=torch.float32)
        buf126 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_26.run(buf124, squeeze_49, buf125, buf126, 128, 13, grid=grid(128), stream=stream0)
        buf127 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_27.run(clone_16, buf120, convolution_16, unsqueeze_230, buf125, squeeze_49, buf123, primals_33, buf127, 1568, 128, grid=grid(1568, 128), stream=stream0)
        del buf120
        del clone_16
        del convolution_16
        del primals_33
        del squeeze_49
        del unsqueeze_230
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf128 = aten.convolution_backward(buf127, div_15, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_15
        del primals_73
        buf129 = buf128[0]
        buf130 = buf128[1]
        del buf128
        buf131 = reinterpret_tensor(buf124, (128, 13), (13, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_23.run(clone_15, buf129, buf131, 1664, 121, grid=grid(1664), stream=stream0)
        buf132 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_24.run(buf131, buf132, 128, 13, grid=grid(128), stream=stream0)
        buf133 = reinterpret_tensor(buf131, (128, 13), (1, 128), 0); del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_25.run(clone_15, buf129, convolution_15, unsqueeze_242, buf133, 1664, 121, grid=grid(1664), stream=stream0)
        buf134 = empty((128, ), device='cuda', dtype=torch.float32)
        buf135 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_26.run(buf133, squeeze_46, buf134, buf135, 128, 13, grid=grid(128), stream=stream0)
        buf136 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_27.run(clone_15, buf129, convolution_15, unsqueeze_242, buf134, squeeze_46, buf132, primals_31, buf136, 1568, 128, grid=grid(1568, 128), stream=stream0)
        del buf129
        del clone_15
        del convolution_15
        del primals_31
        del squeeze_46
        del unsqueeze_242
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf137 = aten.convolution_backward(buf136, div_14, primals_72, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False])
        del div_14
        del primals_72
        buf138 = buf137[0]
        buf139 = buf137[1]
        del buf137
        buf140 = reinterpret_tensor(buf133, (128, 13), (13, 1), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_23.run(clone_14, buf138, buf140, 1664, 121, grid=grid(1664), stream=stream0)
        buf141 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_24.run(buf140, buf141, 128, 13, grid=grid(128), stream=stream0)
        buf142 = reinterpret_tensor(buf140, (128, 13), (1, 128), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_25.run(clone_14, buf138, convolution_14, unsqueeze_254, buf142, 1664, 121, grid=grid(1664), stream=stream0)
        buf143 = empty((128, ), device='cuda', dtype=torch.float32)
        buf144 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_26.run(buf142, squeeze_43, buf143, buf144, 128, 13, grid=grid(128), stream=stream0)
        buf145 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_27.run(clone_14, buf138, convolution_14, unsqueeze_254, buf143, squeeze_43, buf141, primals_29, buf145, 1568, 128, grid=grid(1568, 128), stream=stream0)
        del buf138
        del clone_14
        del convolution_14
        del primals_29
        del squeeze_43
        del unsqueeze_254
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf146 = aten.convolution_backward(buf145, div_13, primals_71, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_13
        del primals_71
        buf147 = buf146[0]
        buf148 = buf146[1]
        del buf146
        buf149 = reinterpret_tensor(buf142, (128, 13), (13, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_23.run(clone_13, buf147, buf149, 1664, 121, grid=grid(1664), stream=stream0)
        buf150 = buf143; del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_24.run(buf149, buf150, 128, 13, grid=grid(128), stream=stream0)
        buf151 = reinterpret_tensor(buf149, (128, 13), (1, 128), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_25.run(clone_13, buf147, convolution_13, unsqueeze_266, buf151, 1664, 121, grid=grid(1664), stream=stream0)
        buf152 = empty((128, ), device='cuda', dtype=torch.float32)
        buf153 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_26.run(buf151, squeeze_40, buf152, buf153, 128, 13, grid=grid(128), stream=stream0)
        buf154 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_27.run(clone_13, buf147, convolution_13, unsqueeze_266, buf152, squeeze_40, buf150, primals_27, buf154, 1568, 128, grid=grid(1568, 128), stream=stream0)
        del buf147
        del clone_13
        del convolution_13
        del primals_27
        del squeeze_40
        del unsqueeze_266
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf155 = aten.convolution_backward(buf154, div_12, primals_70, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False])
        del div_12
        del primals_70
        buf156 = buf155[0]
        buf157 = buf155[1]
        del buf155
        buf158 = reinterpret_tensor(buf151, (128, 13), (13, 1), 0); del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_23.run(clone_12, buf156, buf158, 1664, 121, grid=grid(1664), stream=stream0)
        buf159 = buf152; del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_24.run(buf158, buf159, 128, 13, grid=grid(128), stream=stream0)
        buf160 = reinterpret_tensor(buf158, (128, 13), (1, 128), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_25.run(clone_12, buf156, convolution_12, unsqueeze_278, buf160, 1664, 121, grid=grid(1664), stream=stream0)
        buf161 = empty((128, ), device='cuda', dtype=torch.float32)
        buf162 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_26.run(buf160, squeeze_37, buf161, buf162, 128, 13, grid=grid(128), stream=stream0)
        del buf160
        buf163 = buf154; del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_27.run(clone_12, buf156, convolution_12, unsqueeze_278, buf161, squeeze_37, buf159, primals_25, buf163, 1568, 128, grid=grid(1568, 128), stream=stream0)
        del buf156
        del buf161
        del clone_12
        del convolution_12
        del primals_25
        del squeeze_37
        del unsqueeze_278
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf164 = aten.convolution_backward(buf163, div_11, primals_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_11
        del primals_69
        buf165 = buf164[0]
        buf166 = buf164[1]
        del buf164
        buf167 = empty((64, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_28.run(clone_11, buf165, buf167, 832, 121, grid=grid(832), stream=stream0)
        buf168 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_29.run(buf167, buf168, 64, 13, grid=grid(64), stream=stream0)
        buf169 = reinterpret_tensor(buf167, (64, 13), (1, 64), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_30.run(clone_11, buf165, convolution_11, unsqueeze_290, buf169, 832, 121, grid=grid(832), stream=stream0)
        buf170 = empty((64, ), device='cuda', dtype=torch.float32)
        buf171 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_31.run(buf169, squeeze_34, buf170, buf171, 64, 13, grid=grid(64), stream=stream0)
        del buf169
        buf172 = reinterpret_tensor(buf43, (8, 64, 14, 14), (12544, 1, 896, 64), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_32.run(clone_11, buf165, convolution_11, unsqueeze_290, buf170, squeeze_34, buf168, primals_23, buf172, 1568, 64, grid=grid(1568, 64), stream=stream0)
        del buf165
        del clone_11
        del convolution_11
        del primals_23
        del squeeze_34
        del unsqueeze_290
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf173 = aten.convolution_backward(buf172, div_10, primals_68, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
        del buf172
        del div_10
        del primals_68
        buf174 = buf173[0]
        buf175 = buf173[1]
        del buf173
        buf176 = empty((64, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_33.run(clone_10, buf174, buf176, 3136, 128, grid=grid(3136), stream=stream0)
        buf177 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_34.run(buf176, buf177, 64, 49, grid=grid(64), stream=stream0)
        buf178 = reinterpret_tensor(buf176, (64, 49), (1, 64), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_35.run(clone_10, buf174, convolution_10, unsqueeze_302, buf178, 3136, 128, grid=grid(3136), stream=stream0)
        buf179 = empty((64, ), device='cuda', dtype=torch.float32)
        buf180 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_36.run(buf178, squeeze_31, buf179, buf180, 64, 49, grid=grid(64), stream=stream0)
        buf181 = empty_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_37.run(clone_10, buf174, convolution_10, unsqueeze_302, buf179, squeeze_31, buf177, primals_21, buf181, 6272, 64, grid=grid(6272, 64), stream=stream0)
        del buf174
        del clone_10
        del convolution_10
        del primals_21
        del squeeze_31
        del unsqueeze_302
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf182 = aten.convolution_backward(buf181, div_9, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_9
        del primals_67
        buf183 = buf182[0]
        buf184 = buf182[1]
        del buf182
        buf185 = reinterpret_tensor(buf178, (64, 49), (49, 1), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_33.run(clone_9, buf183, buf185, 3136, 128, grid=grid(3136), stream=stream0)
        buf186 = buf179; del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_34.run(buf185, buf186, 64, 49, grid=grid(64), stream=stream0)
        buf187 = reinterpret_tensor(buf185, (64, 49), (1, 64), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_35.run(clone_9, buf183, convolution_9, unsqueeze_314, buf187, 3136, 128, grid=grid(3136), stream=stream0)
        buf188 = empty((64, ), device='cuda', dtype=torch.float32)
        buf189 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_36.run(buf187, squeeze_28, buf188, buf189, 64, 49, grid=grid(64), stream=stream0)
        buf190 = buf181; del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_37.run(clone_9, buf183, convolution_9, unsqueeze_314, buf188, squeeze_28, buf186, primals_19, buf190, 6272, 64, grid=grid(6272, 64), stream=stream0)
        del buf183
        del clone_9
        del convolution_9
        del primals_19
        del squeeze_28
        del unsqueeze_314
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf191 = aten.convolution_backward(buf190, div_8, primals_66, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
        del div_8
        del primals_66
        buf192 = buf191[0]
        buf193 = buf191[1]
        del buf191
        buf194 = reinterpret_tensor(buf187, (64, 49), (49, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_33.run(clone_8, buf192, buf194, 3136, 128, grid=grid(3136), stream=stream0)
        buf195 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_34.run(buf194, buf195, 64, 49, grid=grid(64), stream=stream0)
        buf196 = reinterpret_tensor(buf194, (64, 49), (1, 64), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_35.run(clone_8, buf192, convolution_8, unsqueeze_326, buf196, 3136, 128, grid=grid(3136), stream=stream0)
        buf197 = empty((64, ), device='cuda', dtype=torch.float32)
        buf198 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_36.run(buf196, squeeze_25, buf197, buf198, 64, 49, grid=grid(64), stream=stream0)
        buf199 = buf190; del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_37.run(clone_8, buf192, convolution_8, unsqueeze_326, buf197, squeeze_25, buf195, primals_17, buf199, 6272, 64, grid=grid(6272, 64), stream=stream0)
        del buf192
        del buf197
        del clone_8
        del convolution_8
        del primals_17
        del squeeze_25
        del unsqueeze_326
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf200 = aten.convolution_backward(buf199, div_7, primals_65, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_7
        del primals_65
        buf201 = buf200[0]
        buf202 = buf200[1]
        del buf200
        buf203 = empty((32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_38.run(clone_7, buf201, buf203, 1568, 128, grid=grid(1568), stream=stream0)
        buf204 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_39.run(buf203, buf204, 32, 49, grid=grid(32), stream=stream0)
        buf205 = reinterpret_tensor(buf203, (32, 49), (1, 32), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_40.run(clone_7, buf201, convolution_7, unsqueeze_338, buf205, 1568, 128, grid=grid(1568), stream=stream0)
        buf206 = empty((32, ), device='cuda', dtype=torch.float32)
        buf207 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_41.run(buf205, squeeze_22, buf206, buf207, 32, 49, grid=grid(32), stream=stream0)
        del buf205
        buf208 = reinterpret_tensor(buf163, (8, 32, 28, 28), (25088, 1, 896, 32), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_42.run(clone_7, buf201, convolution_7, unsqueeze_338, buf206, squeeze_22, buf204, primals_15, buf208, 6272, 32, grid=grid(6272, 32), stream=stream0)
        del buf201
        del clone_7
        del convolution_7
        del primals_15
        del squeeze_22
        del unsqueeze_338
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf209 = aten.convolution_backward(buf208, div_6, primals_64, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf208
        del div_6
        del primals_64
        buf210 = buf209[0]
        buf211 = buf209[1]
        del buf209
        buf212 = empty((32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_43.run(clone_6, buf210, buf212, 6272, 128, grid=grid(6272), stream=stream0)
        buf213 = buf206; del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_44.run(buf212, buf213, 32, 196, grid=grid(32), stream=stream0)
        buf214 = reinterpret_tensor(buf212, (32, 196), (1, 32), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_45.run(clone_6, buf210, convolution_6, unsqueeze_350, buf214, 6272, 128, grid=grid(6272), stream=stream0)
        buf215 = empty((32, ), device='cuda', dtype=torch.float32)
        buf216 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_46.run(buf214, squeeze_19, buf215, buf216, 32, 196, grid=grid(32), stream=stream0)
        buf217 = empty_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_47.run(clone_6, buf210, convolution_6, unsqueeze_350, buf215, squeeze_19, buf213, primals_13, buf217, 25088, 32, grid=grid(25088, 32), stream=stream0)
        del buf210
        del clone_6
        del convolution_6
        del primals_13
        del squeeze_19
        del unsqueeze_350
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf218 = aten.convolution_backward(buf217, div_5, primals_63, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_5
        del primals_63
        buf219 = buf218[0]
        buf220 = buf218[1]
        del buf218
        buf221 = reinterpret_tensor(buf214, (32, 196), (196, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_43.run(clone_5, buf219, buf221, 6272, 128, grid=grid(6272), stream=stream0)
        buf222 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_44.run(buf221, buf222, 32, 196, grid=grid(32), stream=stream0)
        buf223 = reinterpret_tensor(buf221, (32, 196), (1, 32), 0); del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_45.run(clone_5, buf219, convolution_5, unsqueeze_362, buf223, 6272, 128, grid=grid(6272), stream=stream0)
        buf224 = empty((32, ), device='cuda', dtype=torch.float32)
        buf225 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_46.run(buf223, squeeze_16, buf224, buf225, 32, 196, grid=grid(32), stream=stream0)
        buf226 = buf217; del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_47.run(clone_5, buf219, convolution_5, unsqueeze_362, buf224, squeeze_16, buf222, primals_11, buf226, 25088, 32, grid=grid(25088, 32), stream=stream0)
        del buf219
        del clone_5
        del convolution_5
        del primals_11
        del squeeze_16
        del unsqueeze_362
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf227 = aten.convolution_backward(buf226, div_4, primals_62, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del div_4
        del primals_62
        buf228 = buf227[0]
        buf229 = buf227[1]
        del buf227
        buf230 = reinterpret_tensor(buf223, (32, 196), (196, 1), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_43.run(clone_4, buf228, buf230, 6272, 128, grid=grid(6272), stream=stream0)
        buf231 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_44.run(buf230, buf231, 32, 196, grid=grid(32), stream=stream0)
        buf232 = reinterpret_tensor(buf230, (32, 196), (1, 32), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_45.run(clone_4, buf228, convolution_4, unsqueeze_374, buf232, 6272, 128, grid=grid(6272), stream=stream0)
        buf233 = empty((32, ), device='cuda', dtype=torch.float32)
        buf234 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_46.run(buf232, squeeze_13, buf233, buf234, 32, 196, grid=grid(32), stream=stream0)
        buf235 = buf226; del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_47.run(clone_4, buf228, convolution_4, unsqueeze_374, buf233, squeeze_13, buf231, primals_9, buf235, 25088, 32, grid=grid(25088, 32), stream=stream0)
        del buf228
        del buf233
        del clone_4
        del convolution_4
        del primals_9
        del squeeze_13
        del unsqueeze_374
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf236 = aten.convolution_backward(buf235, div_3, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_3
        del primals_61
        buf237 = buf236[0]
        buf238 = buf236[1]
        del buf236
        buf239 = reinterpret_tensor(buf196, (16, 196), (196, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_48.run(clone_3, buf237, buf239, 3136, 128, grid=grid(3136), stream=stream0)
        buf240 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_49.run(buf239, buf240, 16, 196, grid=grid(16), stream=stream0)
        buf241 = reinterpret_tensor(buf239, (16, 196), (1, 16), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_50.run(clone_3, buf237, convolution_3, unsqueeze_386, buf241, 3136, 128, grid=grid(3136), stream=stream0)
        buf242 = empty((16, ), device='cuda', dtype=torch.float32)
        buf243 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_51.run(buf241, squeeze_10, buf242, buf243, 16, 196, grid=grid(16), stream=stream0)
        del buf241
        buf244 = reinterpret_tensor(buf199, (8, 16, 56, 56), (50176, 1, 896, 16), 0); del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_52.run(clone_3, buf237, convolution_3, unsqueeze_386, buf242, squeeze_10, buf240, primals_7, buf244, 25088, 16, grid=grid(25088, 16), stream=stream0)
        del buf237
        del clone_3
        del convolution_3
        del primals_7
        del squeeze_10
        del unsqueeze_386
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf245 = aten.convolution_backward(buf244, div_2, primals_60, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False])
        del buf244
        del div_2
        del primals_60
        buf246 = buf245[0]
        buf247 = buf245[1]
        del buf245
        buf248 = empty((16, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_53.run(clone_2, buf246, buf248, 12544, 128, grid=grid(12544), stream=stream0)
        buf249 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_54.run(buf248, buf249, 16, 784, grid=grid(16), stream=stream0)
        buf250 = reinterpret_tensor(buf248, (16, 784), (1, 16), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_55.run(clone_2, buf246, convolution_2, unsqueeze_398, buf250, 12544, 128, grid=grid(12544), stream=stream0)
        buf251 = empty((16, ), device='cuda', dtype=torch.float32)
        buf252 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_56.run(buf250, squeeze_7, buf251, buf252, 16, 784, grid=grid(16), stream=stream0)
        del buf250
        buf253 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_57.run(clone_2, buf246, convolution_2, unsqueeze_398, buf251, squeeze_7, buf249, primals_5, buf253, 100352, 16, grid=grid(100352, 16), stream=stream0)
        del buf246
        del buf251
        del clone_2
        del convolution_2
        del primals_5
        del squeeze_7
        del unsqueeze_398
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf254 = aten.convolution_backward(buf253, div_1, primals_59, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf253
        del div_1
        del primals_59
        buf255 = buf254[0]
        buf256 = buf254[1]
        del buf254
        buf257 = reinterpret_tensor(buf232, (8, 784), (784, 1), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_58.run(clone_1, buf255, buf257, 6272, 128, grid=grid(6272), stream=stream0)
        buf258 = empty((8, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_59.run(buf257, buf258, 8, 784, grid=grid(8), stream=stream0)
        buf259 = reinterpret_tensor(buf257, (8, 784), (1, 8), 0); del buf257  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_60.run(clone_1, buf255, convolution_1, unsqueeze_410, buf259, 6272, 128, grid=grid(6272), stream=stream0)
        buf260 = empty((8, ), device='cuda', dtype=torch.float32)
        buf261 = empty((8, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_61.run(buf259, squeeze_4, buf260, buf261, 8, 784, grid=grid(8), stream=stream0)
        buf262 = reinterpret_tensor(buf235, (8, 8, 112, 112), (100352, 1, 896, 8), 0); del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_62.run(clone_1, buf255, convolution_1, unsqueeze_410, buf260, squeeze_4, buf258, primals_3, buf262, 100352, 8, grid=grid(100352, 8), stream=stream0)
        del buf255
        del clone_1
        del convolution_1
        del primals_3
        del squeeze_4
        del unsqueeze_410
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf263 = aten.convolution_backward(buf262, div, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del div
        del primals_58
        buf264 = buf263[0]
        buf265 = buf263[1]
        del buf263
        buf266 = reinterpret_tensor(buf259, (8, 784), (784, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_58.run(clone, buf264, buf266, 6272, 128, grid=grid(6272), stream=stream0)
        buf267 = buf260; del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_59.run(buf266, buf267, 8, 784, grid=grid(8), stream=stream0)
        buf268 = reinterpret_tensor(buf266, (8, 784), (1, 8), 0); del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_60.run(clone, buf264, convolution, unsqueeze_422, buf268, 6272, 128, grid=grid(6272), stream=stream0)
        buf269 = empty((8, ), device='cuda', dtype=torch.float32)
        buf270 = empty((8, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_61.run(buf268, squeeze_1, buf269, buf270, 8, 784, grid=grid(8), stream=stream0)
        del buf268
        buf271 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_62.run(clone, buf264, convolution, unsqueeze_422, buf269, squeeze_1, buf267, primals_1, buf271, 100352, 8, grid=grid(100352, 8), stream=stream0)
        del buf264
        del buf269
        del clone
        del convolution
        del primals_1
        del squeeze_1
        del unsqueeze_422
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf272 = aten.convolution_backward(buf271, primals_175, primals_57, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf271
        del primals_175
        del primals_57
        buf273 = buf272[1]
        return (buf270, buf267, buf261, buf258, buf252, buf249, buf243, buf240, buf234, buf231, buf225, buf222, buf216, buf213, buf207, buf204, buf198, buf195, buf189, buf186, buf180, buf177, buf171, buf168, buf162, buf159, buf153, buf150, buf144, buf141, buf135, buf132, buf126, buf123, buf117, buf114, buf108, buf105, buf99, buf96, buf90, buf87, buf81, buf78, buf72, buf69, buf63, buf59, buf42, buf39, buf33, buf29, buf12, buf9, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), buf273, buf265, buf256, buf247, buf238, buf229, buf220, buf211, buf202, buf193, buf184, buf175, buf166, buf157, buf148, buf139, buf130, buf121, buf112, buf103, buf94, buf85, buf76, buf67, buf57, buf54, buf52, buf49, buf46, buf37, buf27, buf24, buf22, buf19, buf16, buf7, buf4, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((8, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((16, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((32, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1280, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cuda:0', dtype=torch.float32)
    div = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_1 = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_2 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 16, 56, 56), (50176, 1, 896, 16), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_3 = rand_strided((8, 16, 56, 56), (50176, 1, 896, 16), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((8, 16, 56, 56), (50176, 1, 896, 16), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_4 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_5 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    div_5 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_6 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    div_6 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_7 = rand_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_8 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_9 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_10 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_11 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_12 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_13 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_14 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_15 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_16 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_17 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_18 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_19 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_20 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    div_20 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_21 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_22 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 128, 7, 7), (6272, 1, 896, 128), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_23 = rand_strided((8, 128, 7, 7), (6272, 1, 896, 128), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((8, 128, 7, 7), (6272, 1, 896, 128), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((8, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((8, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    mul_192 = rand_strided((8, 128, 7, 7), (6272, 1, 896, 128), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_24 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_25 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cuda:0', dtype=torch.float32)
    mean_1 = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    mul_209 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_26 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cuda:0', dtype=torch.float32)
    mean_2 = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((8, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_110 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bool)
    unsqueeze_122 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_134 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_1 = rand_strided((8, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bool)
    unsqueeze_146 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_158 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_170 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_182 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_194 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_206 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_218 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_230 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_242 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_254 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_266 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_278 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_290 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_302 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_314 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_326 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_338 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_350 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_362 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_374 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_386 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_398 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_410 = rand_strided((1, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_422 = rand_strided((1, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_83, primals_85, primals_86, primals_87, primals_89, primals_91, primals_92, primals_175, convolution, squeeze_1, clone, div, convolution_1, squeeze_4, clone_1, div_1, convolution_2, squeeze_7, clone_2, div_2, convolution_3, squeeze_10, clone_3, div_3, convolution_4, squeeze_13, clone_4, div_4, convolution_5, squeeze_16, clone_5, div_5, convolution_6, squeeze_19, clone_6, div_6, convolution_7, squeeze_22, clone_7, div_7, convolution_8, squeeze_25, clone_8, div_8, convolution_9, squeeze_28, clone_9, div_9, convolution_10, squeeze_31, clone_10, div_10, convolution_11, squeeze_34, clone_11, div_11, convolution_12, squeeze_37, clone_12, div_12, convolution_13, squeeze_40, clone_13, div_13, convolution_14, squeeze_43, clone_14, div_14, convolution_15, squeeze_46, clone_15, div_15, convolution_16, squeeze_49, clone_16, div_16, convolution_17, squeeze_52, clone_17, div_17, convolution_18, squeeze_55, clone_18, div_18, convolution_19, squeeze_58, clone_19, div_19, convolution_20, squeeze_61, clone_20, div_20, convolution_21, squeeze_64, clone_21, div_21, convolution_22, squeeze_67, clone_22, div_22, convolution_23, squeeze_70, clone_23, div_23, mean, relu, div_24, mul_192, convolution_26, squeeze_73, clone_24, div_25, convolution_27, squeeze_76, clone_25, div_26, mean_1, relu_1, div_27, mul_209, convolution_30, squeeze_79, clone_26, mean_2, convolution_31, view_1, permute_1, unsqueeze_110, bitwise_and, unsqueeze_122, unsqueeze_134, bitwise_and_1, unsqueeze_146, unsqueeze_158, unsqueeze_170, unsqueeze_182, unsqueeze_194, unsqueeze_206, unsqueeze_218, unsqueeze_230, unsqueeze_242, unsqueeze_254, unsqueeze_266, unsqueeze_278, unsqueeze_290, unsqueeze_302, unsqueeze_314, unsqueeze_326, unsqueeze_338, unsqueeze_350, unsqueeze_362, unsqueeze_374, unsqueeze_386, unsqueeze_398, unsqueeze_410, unsqueeze_422, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('lcnet_050', benchmark_compiled_module)
