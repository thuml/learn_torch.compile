
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


# kernel path: /tmp/torchinductor_youkaichao/rl/crlnvobuwdsdomjub2ox7n33hrhj3tty5k6s3md3kenzcif42iqw.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_hardswish_backward_native_batch_norm_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 960
    x1 = (xindex // 960)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp19 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (960*r2) + (94080*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + (x0 + (960*(r2 // 49)) + (1920*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr2 + (x0 + (960*r2) + (94080*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/sd/csdrqc3ynpphsejrtheiml4m2iksjccbmnhkcrm33vhz44dj7vnz.py
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
    size_hints=[1024, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (960*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coeeh4qbfphuzfxvn2fs5cfizaaowccbqy6kulfukv3xy5z2slvz.py
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
    size_hints=[1024, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_hardswish_backward_native_batch_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (960*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kn/cknxvydmslccgg2fxxgm4h64yphidzojoixg7fmzrack7tpm56ec.py
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
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 960
    x2 = (xindex // 47040)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x0 + (960*x2)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (x3), xmask)
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aj/cajqhjmpf77ohqo56td7so4p75prf2vhok64dcbwd7mnumy5elvp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 160
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (7840*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mq/cmq3r62wmip4vaqxfgvlk23hl5frktg2cid74idsese7ly75zczn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 160
    x1 = (xindex // 160)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (7840*(r2 // 49)) + (15680*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (160*r2) + (15680*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7z/c7zzfrfmgctpwwpnpri4ng74h6ttbettbfftg4t2xsftzysdndnu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/f2/cf2bpdrr7uty3ekyy7vor7xr25hxzzpdf3xezshavbfy7oi3siyy.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 160
    y1 = (yindex // 160)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (160*x2) + (7840*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.002551020408163265
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tk/ctkhbgjk4622bccjtg7eagntd6b357mkcul5da6l6trntrb7rtep.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 960
    x1 = (xindex // 960)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (960*r2) + (47040*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/eq/ceqd5rxno63srhnm4wyp4cihr5f3hpsgtwx4g2a24ruz22j2d3sp.py
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
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (960*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ov/covcglmhd37igui62qwlbi4m2m55yohecw324kjmdgevzcnrljha.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]

triton_poi_fused_hardswish_backward_threshold_backward_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_threshold_backward_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
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


# kernel path: /tmp/torchinductor_youkaichao/sx/csx4z5ijtkehvt3gxsvz3l6lnj4mxcenkgw4saljosgs7uxqemd7.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (240*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a7/ca7u6tirpcp3qrtcqxq52bnd2dts7om4o6v5mck4sxapsam2ebc7.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 960
    x1 = (xindex // 960)
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp23 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (960*r2) + (94080*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((49*x0) + (47040*(r2 // 49)) + (94080*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (960*(r2 // 49)) + (1920*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (960*(r2 // 49)) + (1920*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr4 + (x0 + (960*r2) + (94080*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/mp/cmpsm7jobovz25vidneabcvr4yyw2jhmz3pkg5wztqeceikasvdv.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 960
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
    tmp0 = tl.load(in_ptr0 + (x2 + (960*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (49*x2) + (47040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (960*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (960*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x2 + (960*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2 + (960*y3)), tmp35, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbv5aquztpiixnanwth75ox3iyjf4tafgq3yhz6ym4u2m67miti5.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 960
    x1 = (xindex // 960)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (960*r2) + (94080*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((49*x0) + (47040*(r2 // 49)) + (94080*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr2 + (x0 + (960*r2) + (94080*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/47/c4742pfgfiqhgrxzwko6m2twg5x7dlfesra3mig6obnmrpiugjf4.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 960
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
    tmp0 = tl.load(in_ptr0 + (x2 + (960*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (49*x2) + (47040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (960*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (960*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5g/c5g6g2zzpprdafdlxiv6shqkvr2xrdleuhtbah5rln5tevxedin4.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 160
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
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (7840*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (7840*r2)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0 + (160*r3)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp9 = tmp7 - tmp8
    tmp10 = tmp2 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a4/ca4w6zbpoa24r4qd2rcrpfruqjzbvs74r6kkvjh5uscwdgkfqkp5.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 160
    y1 = (yindex // 160)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (160*x2) + (7840*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.002551020408163265
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3z/c3ziakwiqle67juodqsgg4sgpr3s6ny6bnhokqmqa75faykvngip.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 160
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
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (7840*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (7840*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (49*x0) + (7840*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (x0 + (160*r3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp11 = tmp9 - tmp10
    tmp12 = tmp4 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr2 + (x0), tmp18, xmask)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sd/csdslv7fkeiu36b6esjhu7x3n37ntyavezxyp57jiowilvdqh6pe.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 160
    y1 = (yindex // 160)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + (160*x2) + (7840*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fl/cflactst74wwdsgidlbmiif5z6jyt46i6bcpz2qdhqj2feqdheqt.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_23', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 672
    x1 = (xindex // 672)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (672*r2) + (32928*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/3u/c3uquvqf7rxdcyl65txyakpz3txbd2eyiqiuubw6g3r7y6cxevbg.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/px/cpx7e2ia54n6itj4krlrmigyjnqu6oqohwut72cdkngv35ux6vyh.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]

triton_poi_fused_hardswish_backward_threshold_backward_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_threshold_backward_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1344
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


# kernel path: /tmp/torchinductor_youkaichao/6g/c6gk4lpxeyzdniwug2ibbabax6yp65fsznowxa32snr3qfsxhki5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 168
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (168*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tv/ctvn7izeahddmazbiihwa7wqz4ocwga7zihp34wf37efqg45osww.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2688
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 672
    x1 = (xindex // 672)
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp23 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (65856*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((49*x0) + (32928*(r2 // 49)) + (65856*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (672*(r2 // 49)) + (1344*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (672*(r2 // 49)) + (1344*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr4 + (x0 + (672*r2) + (65856*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/tb/ctbwqzcfimtizbfxyddgtpxlnuo43k62yijulu7cqg25y56u74ia.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zi/czipumuggzt7kjv7vfkzzgpuiz2og5hkdlabloeqz2z4fzaf6gfj.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4v/c4vcl6jcqdl4sc7te4vmxyhoopt6jml4mrue5u6eetvn7y7bx3rs.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 672
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
    tmp0 = tl.load(in_ptr0 + (x2 + (672*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (49*x2) + (32928*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (672*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (672*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x2 + (672*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2 + (672*y3)), tmp35, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/um/cumlkfcwcqnsqv7ez3rkvd5l23pclta2oa2ifljlnsgcs44ubtd2.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8736
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
        tmp3 = tl.load(in_ptr0 + (x1 + (672*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x1) + (131712*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ih/cihsyejf7byxzwvz7zi3ucggucqb36iovanknpl4vc64zrpwnn5c.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
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


# kernel path: /tmp/torchinductor_youkaichao/ks/cksekqyd55xfa3foamo4i2hevhrui33jh7jvpm4gebtaop5pkscp.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8736
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 672)
    x0 = xindex % 672
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (672*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x0) + (131712*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp3 / tmp6
        tmp10 = 0.5
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.where(tmp7, tmp12, tmp8)
        tmp14 = 0.0
        tmp15 = tl.where(tmp5, tmp14, tmp13)
        tmp16 = tl.load(in_ptr2 + (x0 + (672*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/3v/c3vpcdw5w77r535mqdebhx6qvdldljtmdaqomehldr5p6agfjbb2.py
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
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/et/cet3ffpy25shkccicceu3akai6atevwmcj5rh7bkmfdbrzzm5p5h.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 672
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
    tmp0 = tl.load(in_ptr0 + (x2 + (672*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (131712*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (672*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (672*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ne/cnen67ahzf7kxwxorwiwb6wpco7zf6knb463ywbxstmww7wkwa27.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 112
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (21952*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vi/cvisaiywc347eu7ve6norocwxkgf5h3zqydrxwqvkoaa3buhko45.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1456
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (21952*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (112*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/47/c47cqt3bie2vm4lwm7f2pfakjbdusafgzqhjks62xqjm6qxcwyce.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 112
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


# kernel path: /tmp/torchinductor_youkaichao/hr/chriup6gsd4u3a4zvimjnzasigr7nc77xma3ctkzmwjxikkatnmy.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (112*x2) + (21952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0006377551020408163
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xa/cxaij4nui3ev24uqdvj72n6uqbbpdkbz2lpgltaov5ir5rwfpdc5.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10752
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 2
    x1 = (xindex // 2) % 672
    x2 = (xindex // 1344)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (98*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (672*r3) + (65856*x0) + (131712*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5i/c5iwgp6ce5bzc3p2ww74yjev5g2nffilc5st7o5lgkkiev65oeew.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_41', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.16666666666666666
    tmp7 = tmp4 * tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yy/cyyaodoaarszwr3gg344ungrwetawiwpxwjobxivmosyrddajh57.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8736
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (672*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x1) + (131712*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x1 + (672*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 * tmp9
        tmp11 = tl.load(in_ptr3 + (x1 + (672*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = 196.0
        tmp13 = tmp11 / tmp12
        tmp14 = tmp10 + tmp13
        tmp15 = tmp3 / tmp6
        tmp16 = 0.5
        tmp17 = tmp15 + tmp16
        tmp18 = tmp14 * tmp17
        tmp19 = tl.where(tmp7, tmp18, tmp14)
        tmp20 = 0.0
        tmp21 = tl.where(tmp5, tmp20, tmp19)
        tmp22 = tl.full(tmp21.shape, 0, tmp21.dtype)
        tmp23 = tl.where(tmp2, tmp21, tmp22)
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mz/cmzyiqqbrqfah6iddnxlpxe5qzk726f5pws4qqxlu5nm356yjdpz.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8736
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 672)
    x0 = xindex % 672
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (672*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x0) + (131712*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (672*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 * tmp9
        tmp11 = tl.load(in_ptr3 + (x0 + (672*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = 196.0
        tmp13 = tmp11 / tmp12
        tmp14 = tmp10 + tmp13
        tmp15 = tmp3 / tmp6
        tmp16 = 0.5
        tmp17 = tmp15 + tmp16
        tmp18 = tmp14 * tmp17
        tmp19 = tl.where(tmp7, tmp18, tmp14)
        tmp20 = 0.0
        tmp21 = tl.where(tmp5, tmp20, tmp19)
        tmp22 = tl.load(in_ptr4 + (x0 + (672*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tmp22 - tmp23
        tmp25 = tmp21 * tmp24
        tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
        tmp27 = tl.where(tmp2, tmp25, tmp26)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask & xmask, tmp30, _tmp29)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3t/c3tl2l4rznr3exikwucnlamexyltqvxi77e7z266kyka4r6uqfbj.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_44', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 672
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
    tmp0 = tl.load(in_ptr0 + (x2 + (672*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (131712*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (672*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (672*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x2 + (672*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp9 = 196.0
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
    tmp23 = 0.0006377551020408163
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
    tl.store(in_out_ptr0 + (x2 + (672*y3)), tmp35, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gm/cgmqmnqnwqtsha5mdv5e5ada22vakksd5ahcarbxrfgw7hfvqj5x.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 112
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (21952*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (21952*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (112*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp11, xmask)
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tmp11 * tmp13
    tl.store(out_ptr2 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pu/cpuqy2eourzerbp3vyjzi6xq4zbxonjmvaaurghu5z4v5hvv2yod.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_46', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (112*x2) + (21952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0006377551020408163
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/is/cis5hge57jiye3tttcu3g5rsmt7v6vlbneszm2lhljbo6fm4bm3f.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 2
    x1 = (xindex // 2) % 480
    x2 = (xindex // 960)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (98*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (480*r3) + (47040*x0) + (94080*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ku/ckunudcybwrk2lrgi34ukl5zmmgu6yji2w53wkpiqz4x3i7dvxym.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_48', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.16666666666666666
    tmp7 = tmp4 * tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2w/c2w7g44pur5y35t7lil2dxy5g4urs5hsz7gv2waq4vxgjkpm5yu5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (480*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q3/cq3rwztxalib6lv5ll6xyqwwvsx4ghbijf7fcrippt7sksfkqpgw.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]

triton_poi_fused_hardswish_backward_threshold_backward_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_threshold_backward_50', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 960
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


# kernel path: /tmp/torchinductor_youkaichao/p2/cp2l76xdopbe7yxtfsqrkad4apdgpqu2mfmpgcvjznzi77j3ssmb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 120
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (120*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pd/cpdyv43yv3cy7qszuzxfepxao5sx2lby5txwxqz73dapxe3xnpmy.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6240
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (480*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x1) + (94080*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x1 + (480*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 * tmp9
        tmp11 = tl.load(in_ptr3 + (x1 + (480*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = 196.0
        tmp13 = tmp11 / tmp12
        tmp14 = tmp10 + tmp13
        tmp15 = tmp3 / tmp6
        tmp16 = 0.5
        tmp17 = tmp15 + tmp16
        tmp18 = tmp14 * tmp17
        tmp19 = tl.where(tmp7, tmp18, tmp14)
        tmp20 = 0.0
        tmp21 = tl.where(tmp5, tmp20, tmp19)
        tmp22 = tl.full(tmp21.shape, 0, tmp21.dtype)
        tmp23 = tl.where(tmp2, tmp21, tmp22)
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jj/cjjowvxh4fgv2rot4cwqwxamzhpenw4womrqxgmvsywrjbqjim2a.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
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


# kernel path: /tmp/torchinductor_youkaichao/tu/ctuop2mc6gdtutzv2j4qtyz5bswa4amgfvj2npq2g34ozsskwfx7.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6240
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 480)
    x0 = xindex % 480
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (480*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x0) + (94080*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (480*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 * tmp9
        tmp11 = tl.load(in_ptr3 + (x0 + (480*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = 196.0
        tmp13 = tmp11 / tmp12
        tmp14 = tmp10 + tmp13
        tmp15 = tmp3 / tmp6
        tmp16 = 0.5
        tmp17 = tmp15 + tmp16
        tmp18 = tmp14 * tmp17
        tmp19 = tl.where(tmp7, tmp18, tmp14)
        tmp20 = 0.0
        tmp21 = tl.where(tmp5, tmp20, tmp19)
        tmp22 = tl.load(in_ptr4 + (x0 + (480*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tmp22 - tmp23
        tmp25 = tmp21 * tmp24
        tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
        tmp27 = tl.where(tmp2, tmp25, tmp26)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask & xmask, tmp30, _tmp29)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qv/cqvcj4szxyisceczflm3evglhncj4brqa35nzzeyr575jb63ykui.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (480*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sp/cspcr4uhiimy45odoxdhmssmstcqsvnrsqv47bdig5pjbcvoobql.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_56', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 480
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
    tmp0 = tl.load(in_ptr0 + (x2 + (480*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (480*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (480*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x2 + (480*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp9 = 196.0
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
    tmp23 = 0.0006377551020408163
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
    tl.store(in_out_ptr0 + (x2 + (480*y3)), tmp35, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uf/cufspnbqmwm6uxmwsj6epurnxnqx7mug4qpkchluuftzetkubz5k.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6240
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
        tmp3 = tl.load(in_ptr0 + (x1 + (480*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x1) + (94080*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/54/c545emvcjmrtrizatxqq4uv2muub4fzjatcnrx3bthyqhlwum7ci.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6240
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 480)
    x0 = xindex % 480
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (480*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x0) + (94080*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp3 / tmp6
        tmp10 = 0.5
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.where(tmp7, tmp12, tmp8)
        tmp14 = 0.0
        tmp15 = tl.where(tmp5, tmp14, tmp13)
        tmp16 = tl.load(in_ptr2 + (x0 + (480*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bd/cbd5odp53qckeiz4ql6gfkwrk6qiqw3i3k5cyjozhv5zjgcqeb6s.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 480
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
    tmp0 = tl.load(in_ptr0 + (x2 + (480*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (480*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (480*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/za/czatkf76rrgiqfnxvwrtnta2qmawm4rnqkw2gaxsvj6zklui6lrm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i7/ci7me6655blpkplaybqljsu7uidxqwbvqjychaywiintod34ms5a.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1040
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (15680*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (80*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/mi/cmivvtsi2k7sskr2t5rofuzmkj3losz4mizmht55vf2pyvgjhbyf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 80
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


# kernel path: /tmp/torchinductor_youkaichao/22/c22njrvsscgck5otr2xtcupd63zewls4c2cmrvr5eqi3xd6lrcun.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 80
    y1 = (yindex // 80)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (80*x2) + (15680*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0006377551020408163
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nk/cnkflqt4iq6n7zkvjnox624bzjyxejihzj4xi7hfng5f6t5vehjl.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2392
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
        tmp3 = tl.load(in_ptr0 + (x1 + (184*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x1) + (36064*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/6f/c6fchr3hzsbszpuvp6dpaqbpgrmfpgxehylxwqvuup6unf7n6skr.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 184
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


# kernel path: /tmp/torchinductor_youkaichao/pp/cppxakr7gaihmnnwducqqyjaesq4vxte3sse2uinulcpzt33fy5y.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2392
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 184)
    x0 = xindex % 184
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (184*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x0) + (36064*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp3 / tmp6
        tmp10 = 0.5
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.where(tmp7, tmp12, tmp8)
        tmp14 = 0.0
        tmp15 = tl.where(tmp5, tmp14, tmp13)
        tmp16 = tl.load(in_ptr2 + (x0 + (184*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/gf/cgf66qvp43iphlz5nuxgs46j4goeshidpikzjdssg3gbezjcwz4v.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 184
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (184*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y3/cy3rn7p4oorqjgge23vo4ndgzo7lk7xi4futwkxlivgkqhhhkowz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 184
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
    tmp0 = tl.load(in_ptr0 + (x2 + (184*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (36064*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (184*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (184*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pn/cpn2juesknqqf3c3nzrxilppjmgllplz4cpjy667ujg2fz5upi5k.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (80*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp11, xmask)
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tmp11 * tmp13
    tl.store(out_ptr2 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nu/cnuonqscwfneolczphw3g4ryhykmffh3tp3bcy6vn4s45okei5ta.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_70 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 80
    y1 = (yindex // 80)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (80*x2) + (15680*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0006377551020408163
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sh/cshauf6bgiw77ksvi2i3jgbjfyqvvvegpktvatwgmz72t53yc7r2.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (80*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jw/cjw3yprudxkfimurq4hy6iawjtxunzqsnzvxl7x7lxk6rdgc465o.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 80
    y1 = (yindex // 80)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (80*x2) + (15680*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q3/cq3dnohj7gwkmdc5k4ud3nxwhnat2drl3jo7noz32i55wqsauyln.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2600
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
        tmp3 = tl.load(in_ptr0 + (x1 + (200*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x1) + (39200*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ws/cwsmwos26f23uv37abt76v3yzebpaxg6tphrckxku2gsahxm6gtf.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_74 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 200
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


# kernel path: /tmp/torchinductor_youkaichao/kp/ckpgqg6vgrdceqn44n2qzswea7uy7tt6nf67f4xbb2dou22boqx5.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2600
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 200)
    x0 = xindex % 200
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (200*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x0) + (39200*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp3 / tmp6
        tmp10 = 0.5
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.where(tmp7, tmp12, tmp8)
        tmp14 = 0.0
        tmp15 = tl.where(tmp5, tmp14, tmp13)
        tmp16 = tl.load(in_ptr2 + (x0 + (200*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/sv/csvm54pqpa4jslq3hsc5r42rr72m2lcfwrfhlgu2ngxpmmeqavjr.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 200
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (200*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mb/cmbhkbuqu3oou5z4nfsm7ndyo6plisebeetqp5tmkeb3nozumypy.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 200
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
    tmp0 = tl.load(in_ptr0 + (x2 + (200*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (39200*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (200*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (200*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dx/cdxbmtm4td4zv3of3poxjidolegmmrujqhdnnsj5q6q4dvbj6edy.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_78 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr4 + (x0 + (80*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp15, xmask)
    tmp17 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tmp15 * tmp17
    tl.store(out_ptr2 + (x0), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/or/corkzczbho267twwr5orr6psf3b3gqugbgtamigp3vk2vnfzc577.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_79 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_79', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 80
    y1 = (yindex // 80)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (80*x2) + (15680*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5g/c5g5i3gcbo223tgorfeabveic2ftotpykyrquohivvobw36l443s.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3120
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
        tmp3 = tl.load(in_ptr0 + (x1 + (240*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x1) + (47040*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/65/c65oh2lmeqbcifjzcxzfrmsgpl6g5dqv2iehoifzqsd2ys75wuix.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
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


# kernel path: /tmp/torchinductor_youkaichao/nc/cncgvjbqefohpuh7ybrifeu4dhjg5ycfscwtsgjjyhqy3su7jj3g.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_82', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3120
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 240)
    x0 = xindex % 240
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (240*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x0) + (47040*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp3 / tmp6
        tmp10 = 0.5
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.where(tmp7, tmp12, tmp8)
        tmp14 = 0.0
        tmp15 = tl.where(tmp5, tmp14, tmp13)
        tmp16 = tl.load(in_ptr2 + (x0 + (240*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/6i/c6isj3ekykozzvn35ibjqh3s5akdqzmjgsr4tcmvoytbtdlbgweo.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_83', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (240*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cp/ccpgz74pkr6mvnp2ft2nw5ifejtshgl2gmopujoayxlyraudiibn.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 240
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
    tmp0 = tl.load(in_ptr0 + (x2 + (240*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (47040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (240*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (240*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ll/clloqynouhyiy3y6exf3mi7outs4g6qdsiiutf7il7kqqvse3edx.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11760
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
        tmp0 = tl.load(in_ptr0 + (x1 + (240*r2) + (30720*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((784*x1) + (188160*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ei/ceiu2chfxhqw3t5wvyqi3rd5d2qu5ztptiuksfopjrkg7k3wdhz4.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_86', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
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


# kernel path: /tmp/torchinductor_youkaichao/4o/c4octlbr65kaykc2xophcztcdanxs3iftbnkny7p2eamsnfrprxv.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11760
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 240
    x1 = (xindex // 240)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((784*x0) + (188160*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/xv/cxvks4abkqoac3kgk2vcuez6nrd5zxqn7d5dr4gnnvqz46vk4zeh.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_88', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (240*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ok/cokbfbibmkka6iwhro6ovf2ivyneliwiife3pp7airoj734a4ivp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_89', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 240
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
    tmp0 = tl.load(in_ptr0 + (x2 + (240*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (784*x2) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (240*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (240*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2p/c2pwqgqbvelk3ofhox3hf4qqfwuk76bdukjed7rm4rzuevztvdsf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_90 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 40
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yj/cyjrwfwgcj6dhfpiisupowhuuc7xtc25hs6emz2h6jwu5ifghfym.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_91 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1960
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (31360*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (40*r2) + (5120*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l4/cl4ariv7o434t2roaz5rlsvofdfeea6erl2ozcm3syienww6w5xa.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_92 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_92', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 40
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


# kernel path: /tmp/torchinductor_youkaichao/hf/chfscmz2ez6ztu2rqcutihy76oaetcpgbucjg7fdrs74lc6v3lmn.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_93 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 40
    y1 = (yindex // 40)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (40*x2) + (31360*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00015943877551020407
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kk/ckkypl27eqhw3xbinuucjeabfowpd723byoo7szisnz4mzysstgd.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_94 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_94', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6720
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 7
    x1 = (xindex // 7) % 120
    x2 = (xindex // 840)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (112*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (120*r3) + (13440*x0) + (94080*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mg/cmge6vzh5nq5w5sculss6irnagzhenweyzi4w5fzrhxan2xxefcu.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_95 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_95', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.16666666666666666
    tmp7 = tmp4 * tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bm/cbmqlwy5dxnpxoaijy4vwktfnnbmnbyei3xti6ebgsa2i6fusjwz.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]

triton_poi_fused_hardswish_backward_threshold_backward_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_threshold_backward_96', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/kg/ckgfb4occsaqd3zskuhcm2thsf5yzrog6rccjndsviryx6sfwj5w.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_97', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/nu/cnuijd7whi5iwtaad5yzwql37vipeemtlzt7kzax3iaewy6dd5qh.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_98 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_98', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5880
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (120*r2) + (15360*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((784*x1) + (94080*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (120*((r2 + (128*x0)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x1 + (120*((r2 + (128*x0)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sm/csmfs2tnewiblrrw337pfvjnhnwz3ynhwfjobkmq7ylntofhgwxp.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_99 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_99', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 120
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


# kernel path: /tmp/torchinductor_youkaichao/co/ccou2ucz2yj5lwryxuuitcb2b642jatl7sedqztewmz7wlvhewns.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_100 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_100', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5880
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 120
    x1 = (xindex // 120)
    tmp12 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (120*r2) + (15360*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((784*x0) + (94080*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (120*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x0 + (120*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr4 + (x0 + (120*r2) + (15360*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = 784.0
        tmp8 = tmp6 / tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp10 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ph/cphey5t5unif5fgk2lkyolys5mgg3sje5p4hdajt3xqgianmqrdg.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_101 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_101', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 120
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (120*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sb/csbydoywylfrynccizzwlwvyezy5jmjjqlrubg7s5jli62kngig4.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_102 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_102', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 120
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
    tmp0 = tl.load(in_ptr0 + (x2 + (120*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (120*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (120*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x2 + (120*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (120*y3)), tmp27, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bw/cbwmunmatho5yyzcbepy2h2shkinmxxq46opxehpvslenet52hb7.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_103 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_103', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5880
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
        tmp0 = tl.load(in_ptr0 + (x1 + (120*r2) + (15360*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((784*x1) + (94080*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gs/cgsqgegatmh2j4bwhau4bdrf7vtxqbbgmq6xknyd65otg5eiisx7.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_104 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_104', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5880
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 120
    x1 = (xindex // 120)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (120*r2) + (15360*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((784*x0) + (94080*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (120*r2) + (15360*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/jo/cjovy6jz4gpj3yuec33jkpotcze4okxzuvdwis6h6j56ejytqvrn.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_105 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_105', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 120
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
    tmp0 = tl.load(in_ptr0 + (x2 + (120*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (120*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (120*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yf/cyfodpgk3a7pr4ks6v25yagqzqw4xzfr74p2aid7bssdszqdsesz.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_106 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_106', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (784*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (40*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp11, xmask)
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tmp11 * tmp13
    tl.store(out_ptr2 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cn/ccnhd3aau2fgx2nwy7rcsla2ji3ihfk55gxpkgkf6vb342w5mgas.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_107 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_107', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 40
    y1 = (yindex // 40)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (40*x2) + (31360*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.00015943877551020407
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jt/cjt3offiih6o7mgggirugppqreirfo53e7zfhh24rivdwrzatuwm.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_108 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_108', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (784*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (784*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (40*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kj/ckjgeqfzsxgjnmg6bqw22zxt6d44q6dg67q2bg6gndo5skxd6ktt.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_109 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_109', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 40
    y1 = (yindex // 40)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + (40*x2) + (31360*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lv/clv3alrng4mp3yilgxsov2w5uyz4xthlozqgqoh3hhifwvhu5njw.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_110', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4032
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 7
    x1 = (xindex // 7) % 72
    x2 = (xindex // 504)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (112*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (72*r3) + (8064*x0) + (56448*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x4/cx4yc6ir3tirh45p2rh7u54eqc64ki5shhv6frewltbj6tlcsill.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_111 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_111', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 576
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.16666666666666666
    tmp7 = tmp4 * tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7lmfdnb56eztchnrbbfctfnwdnbvmx55frsyeignk4etru277vh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_112 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_112', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 72
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (72*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ft/cftsmrshy37xebh774niq6plcaq7me5noevw5cdzzdvo3wg4tw46.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]

triton_poi_fused_hardswish_backward_threshold_backward_113 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_threshold_backward_113', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
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


# kernel path: /tmp/torchinductor_youkaichao/no/cnosfwv7ugldjg7bm7qn4dblrceu4c3xuk7fthyxralw6l6nbknq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_114 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_114', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (24*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zv/czvs4icii6r2ikeyfqaq5qwgvkcerrr6sss52oks4kd22xgbwk7a.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_115 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_115', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3528
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (72*r2) + (9216*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((784*x1) + (56448*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (72*((r2 + (128*x0)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x1 + (72*((r2 + (128*x0)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5p/c5phvllumuyci6on5tk3ufeorik5omhqlvtbuwgj7mvnqng4bdmt.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_116 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_116', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 72
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


# kernel path: /tmp/torchinductor_youkaichao/of/cofuipcbxnrxmnnwaqkdhfmvexih5lxwld75rjdb25ajc3dwg3ni.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_117 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_117', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3528
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 72
    x1 = (xindex // 72)
    tmp12 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (72*r2) + (9216*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((784*x0) + (56448*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (72*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x0 + (72*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr4 + (x0 + (72*r2) + (9216*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 * tmp4
        tmp7 = 784.0
        tmp8 = tmp6 / tmp7
        tmp9 = tmp5 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp10 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yn/cynuxsvnwlh2qgowowuthw5o6bzgy5hgddxewn6lr2krb5ivm3az.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_118 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_118', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 72
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (72*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d6/cd64g3eag3ihyrlnmnvhaslkdkufu6vlqabzjitd6hi33l3oyigt.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_119 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_119', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 72
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
    tmp0 = tl.load(in_ptr0 + (x2 + (72*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (56448*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (72*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (72*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x2 + (72*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (72*y3)), tmp27, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zi/czih3euclnmdperlz7v327wjttfr2sshz7ptbc2mof6sokp4a6gg.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_120 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_120', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14112
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
        tmp0 = tl.load(in_ptr0 + (x1 + (72*r2) + (9216*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x1) + (225792*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3fhm4lhqe6zg4dqjafwdrdj3xvvai5yyzdsleteth4cbfujtqh.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_121 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_121', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 72
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


# kernel path: /tmp/torchinductor_youkaichao/gg/cgguerythinqflh7qim77lohayjknixzgqggasfv7scertamkfnh.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_122 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_122', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14112
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 72
    x1 = (xindex // 72)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (72*r2) + (9216*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (225792*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (72*r2) + (9216*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qr/cqrtbfj37ly47nhqw6wbns7ezenwr6kzd7fsnviftfdkkjczxwi3.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_123 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_123', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
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
        tmp0 = tl.load(in_ptr0 + (x0 + (72*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t2/ct2vtopbshqq63eythypnqhqwlibhmqbz2z2sohwudadykigojxg.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_124 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_124', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 72
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
    tmp0 = tl.load(in_ptr0 + (x2 + (72*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (225792*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (72*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (72*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4c/c4c4wsulwpjqkxyh5u4kjxmsfnmtz36mm2oahmcti3pz6xsmlicc.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_125 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_125', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2o/c2o2odldr3ngyrzosbnn36c777smg7wxknrrqam7m56vt4j3ks7t.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_126 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_126', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (24*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zx/czxyaa5xxg47phtom3rofkc2cqa7vnavqy3w322n6z4bblk6j5zh.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_127 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_127', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x1) + (75264*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (24*r2) + (3072*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wd/cwdbpmf4if6fyxjiqqg4pav24yehnvtwqclary7bhtknykesw256.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_128 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_128', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
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


# kernel path: /tmp/torchinductor_youkaichao/uf/cufzwpewhkxxxzynd4mfi5sdgmwqomcxwlih3qrhxofbb2ad72uw.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_129 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_129', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (24*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 3.985969387755102e-05
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5zcjcf5xjoevejaxe5rf2lnp4wldni2z2uy6qseg646rf4mdzu.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_130 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_130', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (24*r2) + (150528*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r6/cr64vsustzsdwy4mqp3pmcudk2ixlujepzt5sm3hbtezhpi3zvdq.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_131 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_131', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (24*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ka/ckajspd3wavmyamy7arjlfuh7agoavtuunn2m4u4gypqwrvcxcph.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_132 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_132', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (24*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 3.985969387755102e-05
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (3136*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/co/cconhsbpwctkpdoy3nfldymsufo3ukyape776uif7j25cdbuy3qn.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_133 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_133', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
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
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x1) + (200704*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iq/ciqbuz4xzwobgeamwtk4izkpxo63d7llvo4bhg3fgxgxrpwjs3nz.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_134 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_134', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4p/c4p2mrvhpp44iczezfo4dnptylz4vz4hjvpimhuif6xhq4axzncw.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_135 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_135', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
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
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (200704*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7u/c7uhqvdjlcyafplruofln3jw7paorse2ito5abwck53edxj3eako.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_136 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_136', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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


# kernel path: /tmp/torchinductor_youkaichao/jy/cjycvn74voe4hokikhs5uct454oaqgsnevx4kj7y7a5sakhf5ufu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_137 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_137', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3x/c3x3fifqpfeuoff3hfwzurv6j3g3q7prx5n3olohujqhmzzlxmvi.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_138 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_138', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/o3/co3hxqpwogob3z3j6aedf5qfc24ransz353bufsibigmhv7cv73f.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_139 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_139', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ii/ciiqh2rfbmpkhljmdazpftqrthvcx6usonnfsjkxgsm7tzvh5j6p.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_140 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_140', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/nz/cnzasapp4l3fog3pfr3bgwb2kewzaqyrmnvrx7idspy7qtwtz53c.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_141 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_141', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/hp/chphti7eaxekh7hhe2n6mlyv56twkpjoprsz27hh7vwluntsvxck.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_142 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_142', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/6c/c6cgqsa6374zvtakvlc6rijwpaxpew67z3wwxqqpeis564mthqwl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_143 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_143', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 208
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x0)
        tmp1 = tl.full([1, 1], 100352, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((12544*x1) + (200704*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sv/csvrivigmleln7wkia77pck6v3ztdac7exxrjdr47ffcydtepr46.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_144 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_144', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxgdywiuftcp2ni7rvf7tcnuw6bzbpyjyqrdsn44wqk23jtscgu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_145 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_145', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((12544*x1) + (200704*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (16*r2) + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j4/cj4cecen7ac5bfyu3oqsk2qbimbi27ajrsho2qhrhpsrnvn5v7l3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_146 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_146', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dy/cdy6h5spwbeo4xncrrbbtvoea7nypjsrjj7m32252ldjkdzovsfl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_147 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_147', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (16*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 9.964923469387754e-06
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (12544*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7v/c7venyxqirlcnd24ejo2qyro6exzoaz2xji4xxnq4pzm5bm67crv.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_148 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_148', 'mutated_arg_names': []}
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
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (16*r2) + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x1) + (200704*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3flcyh7pl2iqambkvpt5sm5sccanivndmrsqsesbgxigwkgwb2.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_149 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_149', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/gs/cgsekuamyv2y63d6zpnema5efncfl7wzpucpff2jhbouxoqbexll.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_150 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_150', 'mutated_arg_names': []}
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
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r2) + (2048*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x0) + (200704*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (16*r2) + (2048*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7d/c7dzg6dfdnlwsar6hlzb5shk3kdfxvsyackbqy556izekq76thfe.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_151 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_151', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/f2/cf2p4tfbssj7paptc3tfbmq5eykudyhjm5jnvanarakg7wm7e5zc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_152 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_152', 'mutated_arg_names': []},
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
    tmp3 = tl.load(in_ptr1 + (y0 + (12544*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (16*y3)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (16*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7o/c7op7smlryfz5oxhmlxjz7wailc65w2yvuyiwf6zf3ysshub2xiu.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_add_hardswish_backward_native_batch_norm_backward_153 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardswish_backward_native_batch_norm_backward_153', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 208
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x0)
        tmp1 = tl.full([1, 1], 100352, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (16*((r2 + (7720*x0)) % 100352))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((12544*x1) + (200704*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + ((12544*x1) + (200704*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp3 / tmp6
        tmp12 = 0.5
        tmp13 = tmp11 + tmp12
        tmp14 = tmp10 * tmp13
        tmp15 = tl.where(tmp7, tmp14, tmp10)
        tmp16 = 0.0
        tmp17 = tl.where(tmp5, tmp16, tmp15)
        tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
        tmp19 = tl.where(tmp2, tmp17, tmp18)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6o/c6oc2q7yjwizz53ptpheshvtgkqru2kimwcmzpesxpwywt5zk4jt.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_add_hardswish_backward_native_batch_norm_backward_154 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardswish_backward_native_batch_norm_backward_154', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp16 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (16*r2) + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((12544*x1) + (200704*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + ((12544*x1) + (200704*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr3 + (x1 + (16*r2) + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = tmp0 / tmp3
        tmp9 = 0.5
        tmp10 = tmp8 + tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.where(tmp4, tmp11, tmp7)
        tmp13 = 0.0
        tmp14 = tl.where(tmp2, tmp13, tmp12)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp14 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uc/cucy53hb5eqsisefzkbbebb7ps3255hyhco5q34arnhpsbfass7o.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_hardswish_backward_native_batch_norm_backward_155 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_hardswish_backward_native_batch_norm_backward_155', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp6 = tl.load(in_ptr2 + (y0 + (12544*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x2 + (16*y3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp3
    tmp9 = 0.5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tl.where(tmp4, tmp11, tmp7)
    tmp13 = 0.0
    tmp14 = tl.where(tmp2, tmp13, tmp12)
    tmp17 = tmp15 - tmp16
    tmp19 = 9.964923469387754e-06
    tmp20 = tmp18 * tmp19
    tmp22 = tmp21 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp17 * tmp23
    tmp25 = tmp14 - tmp24
    tmp27 = tmp26 * tmp19
    tmp28 = tmp25 - tmp27
    tmp30 = tmp21 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x2 + (16*y3)), tmp31, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_108, primals_110, primals_111, primals_112, primals_113, primals_115, primals_117, primals_118, primals_119, primals_120, primals_122, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_141, primals_143, primals_144, primals_145, primals_146, primals_148, primals_150, primals_151, primals_152, primals_153, primals_155, primals_157, primals_158, primals_159, primals_160, primals_162, primals_164, primals_165, primals_166, primals_167, primals_169, primals_171, primals_172, primals_173, primals_313, convolution, squeeze_1, clone, div, convolution_1, squeeze_4, relu, convolution_2, squeeze_7, add_16, convolution_3, squeeze_10, relu_1, convolution_4, squeeze_13, relu_2, convolution_5, squeeze_16, add_31, convolution_6, squeeze_19, relu_3, convolution_7, squeeze_22, relu_4, convolution_8, squeeze_25, add_47, convolution_9, squeeze_28, relu_5, convolution_10, squeeze_31, relu_6, mean, relu_7, div_1, mul_78, convolution_13, squeeze_34, add_63, convolution_14, squeeze_37, relu_8, convolution_15, squeeze_40, relu_9, mean_1, relu_10, div_2, mul_100, convolution_18, squeeze_43, add_80, convolution_19, squeeze_46, relu_11, convolution_20, squeeze_49, relu_12, mean_2, relu_13, div_3, mul_122, convolution_23, squeeze_52, add_97, convolution_24, squeeze_55, clone_1, div_4, convolution_25, squeeze_58, clone_2, div_5, convolution_26, squeeze_61, add_114, convolution_27, squeeze_64, clone_3, div_6, convolution_28, squeeze_67, clone_4, div_7, convolution_29, squeeze_70, add_132, convolution_30, squeeze_73, clone_5, div_8, convolution_31, squeeze_76, clone_6, div_9, convolution_32, squeeze_79, add_150, convolution_33, squeeze_82, clone_7, div_10, convolution_34, squeeze_85, clone_8, div_11, convolution_35, squeeze_88, add_168, convolution_36, squeeze_91, clone_9, div_12, convolution_37, squeeze_94, clone_10, div_13, mean_3, relu_14, div_14, mul_238, convolution_40, squeeze_97, add_186, convolution_41, squeeze_100, clone_11, div_15, convolution_42, squeeze_103, clone_12, div_16, mean_4, relu_15, div_17, mul_262, convolution_45, squeeze_106, add_205, convolution_46, squeeze_109, clone_13, div_18, convolution_47, squeeze_112, clone_14, div_19, mean_5, relu_16, div_20, mul_286, convolution_50, squeeze_115, add_223, convolution_51, squeeze_118, clone_15, div_21, convolution_52, squeeze_121, clone_16, div_22, mean_6, relu_17, div_23, mul_310, convolution_55, squeeze_124, add_242, convolution_56, squeeze_127, clone_17, div_24, convolution_57, squeeze_130, clone_18, div_25, mean_7, relu_18, div_26, mul_334, convolution_60, squeeze_133, add_261, convolution_61, squeeze_136, clone_19, mean_8, convolution_62, view_1, permute_1, unsqueeze_186, unsqueeze_198, bitwise_and, unsqueeze_210, unsqueeze_222, unsqueeze_234, bitwise_and_1, unsqueeze_246, unsqueeze_258, unsqueeze_270, bitwise_and_2, unsqueeze_282, unsqueeze_294, unsqueeze_306, bitwise_and_3, unsqueeze_318, unsqueeze_330, unsqueeze_342, bitwise_and_4, unsqueeze_354, unsqueeze_366, unsqueeze_378, unsqueeze_390, unsqueeze_402, unsqueeze_414, unsqueeze_426, unsqueeze_438, unsqueeze_450, unsqueeze_462, unsqueeze_474, unsqueeze_486, unsqueeze_498, unsqueeze_510, unsqueeze_522, bitwise_and_5, unsqueeze_534, unsqueeze_546, unsqueeze_558, bitwise_and_6, unsqueeze_570, unsqueeze_582, unsqueeze_594, bitwise_and_7, unsqueeze_606, unsqueeze_618, unsqueeze_630, unsqueeze_642, unsqueeze_654, unsqueeze_666, unsqueeze_678, unsqueeze_690, unsqueeze_702, unsqueeze_714, unsqueeze_726, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (16, ), (1, ))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_11, (24, ), (1, ))
    assert_size_stride(primals_13, (72, ), (1, ))
    assert_size_stride(primals_15, (72, ), (1, ))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_19, (72, ), (1, ))
    assert_size_stride(primals_21, (72, ), (1, ))
    assert_size_stride(primals_23, (40, ), (1, ))
    assert_size_stride(primals_25, (120, ), (1, ))
    assert_size_stride(primals_27, (120, ), (1, ))
    assert_size_stride(primals_29, (40, ), (1, ))
    assert_size_stride(primals_31, (120, ), (1, ))
    assert_size_stride(primals_33, (120, ), (1, ))
    assert_size_stride(primals_35, (40, ), (1, ))
    assert_size_stride(primals_37, (240, ), (1, ))
    assert_size_stride(primals_39, (240, ), (1, ))
    assert_size_stride(primals_41, (80, ), (1, ))
    assert_size_stride(primals_43, (200, ), (1, ))
    assert_size_stride(primals_45, (200, ), (1, ))
    assert_size_stride(primals_47, (80, ), (1, ))
    assert_size_stride(primals_49, (184, ), (1, ))
    assert_size_stride(primals_51, (184, ), (1, ))
    assert_size_stride(primals_53, (80, ), (1, ))
    assert_size_stride(primals_55, (184, ), (1, ))
    assert_size_stride(primals_57, (184, ), (1, ))
    assert_size_stride(primals_59, (80, ), (1, ))
    assert_size_stride(primals_61, (480, ), (1, ))
    assert_size_stride(primals_63, (480, ), (1, ))
    assert_size_stride(primals_65, (112, ), (1, ))
    assert_size_stride(primals_67, (672, ), (1, ))
    assert_size_stride(primals_69, (672, ), (1, ))
    assert_size_stride(primals_71, (112, ), (1, ))
    assert_size_stride(primals_73, (672, ), (1, ))
    assert_size_stride(primals_75, (672, ), (1, ))
    assert_size_stride(primals_77, (160, ), (1, ))
    assert_size_stride(primals_79, (960, ), (1, ))
    assert_size_stride(primals_81, (960, ), (1, ))
    assert_size_stride(primals_83, (160, ), (1, ))
    assert_size_stride(primals_85, (960, ), (1, ))
    assert_size_stride(primals_87, (960, ), (1, ))
    assert_size_stride(primals_89, (160, ), (1, ))
    assert_size_stride(primals_91, (960, ), (1, ))
    assert_size_stride(primals_95, (16, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_96, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_97, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_98, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_99, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_100, (24, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_101, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_102, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_103, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_104, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_105, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_106, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_108, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_110, (40, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_111, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_112, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_113, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_115, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_117, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_118, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_119, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_120, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_122, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_124, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_125, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_126, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_127, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_128, (200, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_129, (200, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_130, (80, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_131, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_132, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_133, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_134, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_135, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_136, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_137, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_138, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_139, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_141, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_143, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_144, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_145, (672, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_146, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_148, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_150, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_151, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_152, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_153, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_155, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_157, (160, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_158, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_159, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_160, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_162, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_164, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_165, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_166, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_167, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_169, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_171, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_172, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_173, (1280, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_313, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_1, (16, ), (1, ))
    assert_size_stride(clone, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(div, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_1, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_4, (16, ), (1, ))
    assert_size_stride(relu, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_7, (16, ), (1, ))
    assert_size_stride(add_16, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_3, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_10, (64, ), (1, ))
    assert_size_stride(relu_1, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(convolution_4, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_13, (64, ), (1, ))
    assert_size_stride(relu_2, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_5, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_16, (24, ), (1, ))
    assert_size_stride(add_31, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_6, (8, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(squeeze_19, (72, ), (1, ))
    assert_size_stride(relu_3, (8, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_7, (8, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(squeeze_22, (72, ), (1, ))
    assert_size_stride(relu_4, (8, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_8, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_25, (24, ), (1, ))
    assert_size_stride(add_47, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_9, (8, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(squeeze_28, (72, ), (1, ))
    assert_size_stride(relu_5, (8, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_10, (8, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(squeeze_31, (72, ), (1, ))
    assert_size_stride(relu_6, (8, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(mean, (8, 72, 1, 1), (72, 1, 72, 72))
    assert_size_stride(relu_7, (8, 24, 1, 1), (24, 1, 24, 24))
    assert_size_stride(div_1, (8, 72, 1, 1), (72, 1, 72, 72))
    assert_size_stride(mul_78, (8, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(convolution_13, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_34, (40, ), (1, ))
    assert_size_stride(add_63, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_14, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_37, (120, ), (1, ))
    assert_size_stride(relu_8, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_15, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_40, (120, ), (1, ))
    assert_size_stride(relu_9, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean_1, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(relu_10, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_2, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_100, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_18, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_43, (40, ), (1, ))
    assert_size_stride(add_80, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_19, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_46, (120, ), (1, ))
    assert_size_stride(relu_11, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_20, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_49, (120, ), (1, ))
    assert_size_stride(relu_12, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean_2, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(relu_13, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_3, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_122, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_23, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_52, (40, ), (1, ))
    assert_size_stride(add_97, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_24, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(squeeze_55, (240, ), (1, ))
    assert_size_stride(clone_1, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(div_4, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(convolution_25, (8, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(squeeze_58, (240, ), (1, ))
    assert_size_stride(clone_2, (8, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(div_5, (8, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(convolution_26, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(squeeze_61, (80, ), (1, ))
    assert_size_stride(add_114, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_27, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(squeeze_64, (200, ), (1, ))
    assert_size_stride(clone_3, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(div_6, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(convolution_28, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(squeeze_67, (200, ), (1, ))
    assert_size_stride(clone_4, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(div_7, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(convolution_29, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(squeeze_70, (80, ), (1, ))
    assert_size_stride(add_132, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_30, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(squeeze_73, (184, ), (1, ))
    assert_size_stride(clone_5, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(div_8, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(convolution_31, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(squeeze_76, (184, ), (1, ))
    assert_size_stride(clone_6, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(div_9, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(convolution_32, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(squeeze_79, (80, ), (1, ))
    assert_size_stride(add_150, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_33, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(squeeze_82, (184, ), (1, ))
    assert_size_stride(clone_7, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(div_10, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(convolution_34, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(squeeze_85, (184, ), (1, ))
    assert_size_stride(clone_8, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(div_11, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(convolution_35, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(squeeze_88, (80, ), (1, ))
    assert_size_stride(add_168, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_36, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_91, (480, ), (1, ))
    assert_size_stride(clone_9, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(div_12, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_37, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_94, (480, ), (1, ))
    assert_size_stride(clone_10, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(div_13, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_3, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(relu_14, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(div_14, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_238, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_40, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(squeeze_97, (112, ), (1, ))
    assert_size_stride(add_186, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_41, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(squeeze_100, (672, ), (1, ))
    assert_size_stride(clone_11, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(div_15, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_42, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(squeeze_103, (672, ), (1, ))
    assert_size_stride(clone_12, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(div_16, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(mean_4, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(relu_15, (8, 168, 1, 1), (168, 1, 168, 168))
    assert_size_stride(div_17, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_262, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_45, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(squeeze_106, (112, ), (1, ))
    assert_size_stride(add_205, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_46, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(squeeze_109, (672, ), (1, ))
    assert_size_stride(clone_13, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(div_18, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_47, (8, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(squeeze_112, (672, ), (1, ))
    assert_size_stride(clone_14, (8, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(div_19, (8, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(mean_5, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(relu_16, (8, 168, 1, 1), (168, 1, 168, 168))
    assert_size_stride(div_20, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_286, (8, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(convolution_50, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(squeeze_115, (160, ), (1, ))
    assert_size_stride(add_223, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_51, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_118, (960, ), (1, ))
    assert_size_stride(clone_15, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(div_21, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_52, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_121, (960, ), (1, ))
    assert_size_stride(clone_16, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(div_22, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(mean_6, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(relu_17, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(div_23, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(mul_310, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_55, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(squeeze_124, (160, ), (1, ))
    assert_size_stride(add_242, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_56, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_127, (960, ), (1, ))
    assert_size_stride(clone_17, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(div_24, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_57, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_130, (960, ), (1, ))
    assert_size_stride(clone_18, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(div_25, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(mean_7, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(relu_18, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(div_26, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(mul_334, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_60, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(squeeze_133, (160, ), (1, ))
    assert_size_stride(add_261, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_61, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_136, (960, ), (1, ))
    assert_size_stride(clone_19, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(mean_8, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(convolution_62, (8, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(view_1, (8, 1280), (1280, 1))
    assert_size_stride(permute_1, (1000, 1280), (1280, 1))
    assert_size_stride(unsqueeze_186, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_198, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(bitwise_and, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(unsqueeze_210, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_222, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_234, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(bitwise_and_1, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(unsqueeze_246, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_270, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(bitwise_and_2, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(unsqueeze_282, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_294, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_306, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(bitwise_and_3, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(unsqueeze_318, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_342, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(bitwise_and_4, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(unsqueeze_354, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_366, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_378, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_390, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(unsqueeze_414, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(unsqueeze_438, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(unsqueeze_450, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_462, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_486, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_498, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(unsqueeze_510, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(unsqueeze_522, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(bitwise_and_5, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(unsqueeze_534, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_546, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_558, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(bitwise_and_6, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(unsqueeze_570, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_582, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_594, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(bitwise_and_7, (8, 72, 1, 1), (72, 1, 72, 72))
    assert_size_stride(unsqueeze_606, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_618, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_630, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_642, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_654, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_666, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_678, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_690, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_702, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_714, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_726, (1, 16, 1, 1), (16, 1, 1, 1))
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
        triton_poi_fused_hardswish_backward_1.run(buf3, convolution_62, 10240, grid=grid(10240), stream=stream0)
        del convolution_62
        buf4 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_2.run(buf3, buf4, 1280, 8, grid=grid(1280), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf5 = aten.convolution_backward(buf3, mean_8, primals_173, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf3
        del mean_8
        del primals_173
        buf6 = buf5[0]
        buf7 = buf5[1]
        del buf5
        buf8 = empty_strided((960, 4), (1, 960), device='cuda', dtype=torch.float32)
        buf10 = empty_strided((960, 4), (1, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_div_hardswish_backward_native_batch_norm_backward_3.run(clone_19, buf6, convolution_61, unsqueeze_186, buf8, buf10, 3840, 98, grid=grid(3840), stream=stream0)
        buf9 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4.run(buf8, buf9, 960, 4, grid=grid(960), stream=stream0)
        buf11 = empty((960, ), device='cuda', dtype=torch.float32)
        buf12 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_5.run(buf10, squeeze_136, buf11, buf12, 960, 4, grid=grid(960), stream=stream0)
        buf13 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_6.run(clone_19, buf6, convolution_61, unsqueeze_186, buf11, squeeze_136, buf9, primals_91, buf13, 376320, grid=grid(376320), stream=stream0)
        del clone_19
        del convolution_61
        del primals_91
        del squeeze_136
        del unsqueeze_186
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf14 = aten.convolution_backward(buf13, add_261, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_261
        del primals_172
        buf15 = buf14[0]
        buf16 = buf14[1]
        del buf14
        buf17 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_7.run(buf15, buf17, 160, 392, grid=grid(160), stream=stream0)
        buf18 = empty_strided((160, 4), (1, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_8.run(buf15, convolution_60, unsqueeze_198, buf18, 640, 98, grid=grid(640), stream=stream0)
        buf19 = empty((160, ), device='cuda', dtype=torch.float32)
        buf20 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_9.run(buf18, squeeze_133, buf19, buf20, 160, 4, grid=grid(160), stream=stream0)
        del buf18
        buf21 = empty((8, 160, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_10.run(buf15, convolution_60, unsqueeze_198, buf19, squeeze_133, buf17, primals_89, buf21, 1280, 49, grid=grid(1280, 49), stream=stream0)
        del convolution_60
        del primals_89
        del squeeze_133
        del unsqueeze_198
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf22 = aten.convolution_backward(buf21, mul_334, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_334
        del primals_171
        buf23 = buf22[0]
        buf24 = buf22[1]
        del buf22
        buf25 = reinterpret_tensor(buf6, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf6  # reuse
        buf26 = reinterpret_tensor(buf25, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_11.run(buf26, buf23, div_25, bitwise_and, 7680, 49, grid=grid(7680), stream=stream0)
        del bitwise_and
        del div_25
        buf27 = buf11; del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf26, buf27, 960, 8, grid=grid(960), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf28 = aten.convolution_backward(buf26, relu_18, primals_169, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf26
        del primals_169
        buf29 = buf28[0]
        buf30 = buf28[1]
        del buf28
        buf31 = buf29; del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]
        triton_poi_fused_hardswish_backward_threshold_backward_13.run(buf31, relu_18, 1920, grid=grid(1920), stream=stream0)
        del relu_18
        buf32 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_14.run(buf31, buf32, 240, 8, grid=grid(240), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf33 = aten.convolution_backward(buf31, mean_7, primals_167, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf31
        del mean_7
        del primals_167
        buf34 = buf33[0]
        buf35 = buf33[1]
        del buf33
        buf36 = buf10; del buf10  # reuse
        buf38 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_15.run(clone_18, buf23, div_26, buf34, convolution_57, unsqueeze_210, buf36, buf38, 3840, 98, grid=grid(3840), stream=stream0)
        buf37 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4.run(buf36, buf37, 960, 4, grid=grid(960), stream=stream0)
        buf39 = empty((960, ), device='cuda', dtype=torch.float32)
        buf41 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_5.run(buf38, squeeze_130, buf39, buf41, 960, 4, grid=grid(960), stream=stream0)
        buf40 = buf13; del buf13  # reuse
        buf42 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_16.run(buf42, clone_18, buf23, div_26, buf34, convolution_57, unsqueeze_210, buf39, squeeze_130, buf37, primals_87, 392, 960, grid=grid(392, 960), stream=stream0)
        del buf23
        del clone_18
        del convolution_57
        del div_26
        del primals_87
        del squeeze_130
        del unsqueeze_210
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf43 = aten.convolution_backward(buf42, div_24, primals_166, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 960, [True, True, False])
        del div_24
        del primals_166
        buf44 = buf43[0]
        buf45 = buf43[1]
        del buf43
        buf46 = buf38; del buf38  # reuse
        buf48 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_17.run(clone_17, buf44, convolution_56, unsqueeze_222, buf46, buf48, 3840, 98, grid=grid(3840), stream=stream0)
        buf47 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4.run(buf46, buf47, 960, 4, grid=grid(960), stream=stream0)
        buf49 = empty((960, ), device='cuda', dtype=torch.float32)
        buf50 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_5.run(buf48, squeeze_127, buf49, buf50, 960, 4, grid=grid(960), stream=stream0)
        buf51 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_18.run(clone_17, buf44, convolution_56, unsqueeze_222, buf49, squeeze_127, buf47, primals_85, buf51, 392, 960, grid=grid(392, 960), stream=stream0)
        del buf44
        del clone_17
        del convolution_56
        del primals_85
        del squeeze_127
        del unsqueeze_222
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf52 = aten.convolution_backward(buf51, add_242, primals_165, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_242
        del primals_165
        buf53 = buf52[0]
        buf54 = buf52[1]
        del buf52
        buf55 = buf19; del buf19  # reuse
        buf56 = empty((160, ), device='cuda', dtype=torch.float32)
        buf57 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_19.run(buf15, buf53, convolution_55, unsqueeze_234, squeeze_124, buf55, buf56, buf57, 160, 392, grid=grid(160), stream=stream0)
        buf58 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_20.run(buf15, buf53, convolution_55, unsqueeze_234, buf56, squeeze_124, buf55, primals_83, buf58, 1280, 49, grid=grid(1280, 49), stream=stream0)
        del convolution_55
        del primals_83
        del squeeze_124
        del unsqueeze_234
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf59 = aten.convolution_backward(buf58, mul_310, primals_164, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf58
        del mul_310
        del primals_164
        buf60 = buf59[0]
        buf61 = buf59[1]
        del buf59
        buf62 = reinterpret_tensor(buf34, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf34  # reuse
        buf63 = reinterpret_tensor(buf62, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_11.run(buf63, buf60, div_22, bitwise_and_1, 7680, 49, grid=grid(7680), stream=stream0)
        del bitwise_and_1
        del div_22
        buf64 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf63, buf64, 960, 8, grid=grid(960), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf65 = aten.convolution_backward(buf63, relu_17, primals_162, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf63
        del primals_162
        buf66 = buf65[0]
        buf67 = buf65[1]
        del buf65
        buf68 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]
        triton_poi_fused_hardswish_backward_threshold_backward_13.run(buf68, relu_17, 1920, grid=grid(1920), stream=stream0)
        del relu_17
        buf69 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_14.run(buf68, buf69, 240, 8, grid=grid(240), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf70 = aten.convolution_backward(buf68, mean_6, primals_160, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf68
        del mean_6
        del primals_160
        buf71 = buf70[0]
        buf72 = buf70[1]
        del buf70
        buf73 = buf48; del buf48  # reuse
        buf75 = buf46; del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_15.run(clone_16, buf60, div_23, buf71, convolution_52, unsqueeze_246, buf73, buf75, 3840, 98, grid=grid(3840), stream=stream0)
        buf74 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4.run(buf73, buf74, 960, 4, grid=grid(960), stream=stream0)
        buf76 = empty((960, ), device='cuda', dtype=torch.float32)
        buf78 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_5.run(buf75, squeeze_121, buf76, buf78, 960, 4, grid=grid(960), stream=stream0)
        buf77 = buf51; del buf51  # reuse
        buf79 = buf77; del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_16.run(buf79, clone_16, buf60, div_23, buf71, convolution_52, unsqueeze_246, buf76, squeeze_121, buf74, primals_81, 392, 960, grid=grid(392, 960), stream=stream0)
        del buf60
        del clone_16
        del convolution_52
        del div_23
        del primals_81
        del squeeze_121
        del unsqueeze_246
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf80 = aten.convolution_backward(buf79, div_21, primals_159, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 960, [True, True, False])
        del div_21
        del primals_159
        buf81 = buf80[0]
        buf82 = buf80[1]
        del buf80
        buf83 = buf75; del buf75  # reuse
        buf85 = buf73; del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_17.run(clone_15, buf81, convolution_51, unsqueeze_258, buf83, buf85, 3840, 98, grid=grid(3840), stream=stream0)
        buf84 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4.run(buf83, buf84, 960, 4, grid=grid(960), stream=stream0)
        del buf83
        buf86 = empty((960, ), device='cuda', dtype=torch.float32)
        buf87 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_5.run(buf85, squeeze_118, buf86, buf87, 960, 4, grid=grid(960), stream=stream0)
        buf88 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_18.run(clone_15, buf81, convolution_51, unsqueeze_258, buf86, squeeze_118, buf84, primals_79, buf88, 392, 960, grid=grid(392, 960), stream=stream0)
        del buf81
        del buf86
        del clone_15
        del convolution_51
        del primals_79
        del squeeze_118
        del unsqueeze_258
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf89 = aten.convolution_backward(buf88, add_223, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_223
        del primals_158
        buf90 = buf89[0]
        buf91 = buf89[1]
        del buf89
        buf92 = buf56; del buf56  # reuse
        buf93 = empty((160, ), device='cuda', dtype=torch.float32)
        buf95 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_21.run(buf15, buf53, buf90, convolution_50, unsqueeze_270, squeeze_115, buf92, buf93, buf95, 160, 392, grid=grid(160), stream=stream0)
        buf94 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_22.run(buf94, buf53, buf90, convolution_50, unsqueeze_270, buf93, squeeze_115, buf92, primals_77, 1280, 49, grid=grid(1280, 49), stream=stream0)
        del buf53
        del buf90
        del buf93
        del convolution_50
        del primals_77
        del squeeze_115
        del unsqueeze_270
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf96 = aten.convolution_backward(buf94, mul_286, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf94
        del mul_286
        del primals_157
        buf97 = buf96[0]
        buf98 = buf96[1]
        del buf96
        buf99 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cuda', dtype=torch.float32)
        buf100 = reinterpret_tensor(buf99, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_23.run(buf100, buf97, div_19, bitwise_and_2, 5376, 49, grid=grid(5376), stream=stream0)
        del bitwise_and_2
        del div_19
        buf101 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_24.run(buf100, buf101, 672, 8, grid=grid(672), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf102 = aten.convolution_backward(buf100, relu_16, primals_155, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf100
        del primals_155
        buf103 = buf102[0]
        buf104 = buf102[1]
        del buf102
        buf105 = buf103; del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]
        triton_poi_fused_hardswish_backward_threshold_backward_25.run(buf105, relu_16, 1344, grid=grid(1344), stream=stream0)
        del relu_16
        buf106 = empty((168, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_26.run(buf105, buf106, 168, 8, grid=grid(168), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf107 = aten.convolution_backward(buf105, mean_5, primals_153, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf105
        del mean_5
        del primals_153
        buf108 = buf107[0]
        buf109 = buf107[1]
        del buf107
        buf110 = empty_strided((672, 4), (1, 672), device='cuda', dtype=torch.float32)
        buf112 = empty_strided((672, 4), (1, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_27.run(clone_14, buf97, div_20, buf108, convolution_47, unsqueeze_282, buf110, buf112, 2688, 98, grid=grid(2688), stream=stream0)
        buf111 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_28.run(buf110, buf111, 672, 4, grid=grid(672), stream=stream0)
        del buf110
        buf113 = empty((672, ), device='cuda', dtype=torch.float32)
        buf115 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_29.run(buf112, squeeze_112, buf113, buf115, 672, 4, grid=grid(672), stream=stream0)
        del buf112
        buf114 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        buf116 = buf114; del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_30.run(buf116, clone_14, buf97, div_20, buf108, convolution_47, unsqueeze_282, buf113, squeeze_112, buf111, primals_75, 392, 672, grid=grid(392, 672), stream=stream0)
        del buf97
        del clone_14
        del convolution_47
        del div_20
        del primals_75
        del squeeze_112
        del unsqueeze_282
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf117 = aten.convolution_backward(buf116, div_18, primals_152, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
        del buf116
        del div_18
        del primals_152
        buf118 = buf117[0]
        buf119 = buf117[1]
        del buf117
        buf120 = empty((672, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_31.run(clone_13, buf118, buf120, 8736, 121, grid=grid(8736), stream=stream0)
        buf121 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_32.run(buf120, buf121, 672, 13, grid=grid(672), stream=stream0)
        buf122 = reinterpret_tensor(buf120, (672, 13), (1, 672), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_33.run(clone_13, buf118, convolution_46, unsqueeze_294, buf122, 8736, 121, grid=grid(8736), stream=stream0)
        buf123 = empty((672, ), device='cuda', dtype=torch.float32)
        buf124 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_34.run(buf122, squeeze_109, buf123, buf124, 672, 13, grid=grid(672), stream=stream0)
        buf125 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_35.run(clone_13, buf118, convolution_46, unsqueeze_294, buf123, squeeze_109, buf121, primals_73, buf125, 1568, 672, grid=grid(1568, 672), stream=stream0)
        del buf118
        del clone_13
        del convolution_46
        del primals_73
        del squeeze_109
        del unsqueeze_294
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf126 = aten.convolution_backward(buf125, add_205, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_205
        del primals_151
        buf127 = buf126[0]
        buf128 = buf126[1]
        del buf126
        buf129 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_36.run(buf127, buf129, 112, 1568, grid=grid(112), stream=stream0)
        buf130 = empty((112, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_37.run(buf127, convolution_45, unsqueeze_306, buf130, 1456, 121, grid=grid(1456), stream=stream0)
        buf131 = empty((112, ), device='cuda', dtype=torch.float32)
        buf132 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_38.run(buf130, squeeze_106, buf131, buf132, 112, 13, grid=grid(112), stream=stream0)
        del buf130
        buf133 = empty((8, 112, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_39.run(buf127, convolution_45, unsqueeze_306, buf131, squeeze_106, buf129, primals_71, buf133, 896, 196, grid=grid(896, 196), stream=stream0)
        del convolution_45
        del primals_71
        del squeeze_106
        del unsqueeze_306
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf134 = aten.convolution_backward(buf133, mul_262, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf133
        del mul_262
        del primals_150
        buf135 = buf134[0]
        buf136 = buf134[1]
        del buf134
        buf137 = empty_strided((8, 672, 1, 1, 2), (1344, 2, 10752, 10752, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_40.run(buf135, div_16, buf137, 10752, 98, grid=grid(10752), stream=stream0)
        del div_16
        buf138 = reinterpret_tensor(buf108, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf108  # reuse
        buf139 = reinterpret_tensor(buf138, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_41.run(buf139, buf137, bitwise_and_3, 5376, 2, grid=grid(5376), stream=stream0)
        del bitwise_and_3
        del buf137
        buf140 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_24.run(buf139, buf140, 672, 8, grid=grid(672), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf141 = aten.convolution_backward(buf139, relu_15, primals_148, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf139
        del primals_148
        buf142 = buf141[0]
        buf143 = buf141[1]
        del buf141
        buf144 = buf142; del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]
        triton_poi_fused_hardswish_backward_threshold_backward_25.run(buf144, relu_15, 1344, grid=grid(1344), stream=stream0)
        del relu_15
        buf145 = empty((168, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_26.run(buf144, buf145, 168, 8, grid=grid(168), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf146 = aten.convolution_backward(buf144, mean_4, primals_146, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf144
        del mean_4
        del primals_146
        buf147 = buf146[0]
        buf148 = buf146[1]
        del buf146
        buf149 = reinterpret_tensor(buf122, (672, 13), (13, 1), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_42.run(clone_12, buf135, div_17, buf147, buf149, 8736, 121, grid=grid(8736), stream=stream0)
        buf150 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_32.run(buf149, buf150, 672, 13, grid=grid(672), stream=stream0)
        buf151 = reinterpret_tensor(buf149, (672, 13), (1, 672), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_43.run(clone_12, buf135, div_17, buf147, convolution_42, unsqueeze_318, buf151, 8736, 121, grid=grid(8736), stream=stream0)
        buf152 = empty((672, ), device='cuda', dtype=torch.float32)
        buf154 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_34.run(buf151, squeeze_103, buf152, buf154, 672, 13, grid=grid(672), stream=stream0)
        buf153 = buf125; del buf125  # reuse
        buf155 = buf153; del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_44.run(buf155, clone_12, buf135, div_17, buf147, convolution_42, unsqueeze_318, buf152, squeeze_103, buf150, primals_69, 1568, 672, grid=grid(1568, 672), stream=stream0)
        del buf135
        del buf147
        del clone_12
        del convolution_42
        del div_17
        del primals_69
        del squeeze_103
        del unsqueeze_318
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf156 = aten.convolution_backward(buf155, div_15, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 672, [True, True, False])
        del div_15
        del primals_145
        buf157 = buf156[0]
        buf158 = buf156[1]
        del buf156
        buf159 = reinterpret_tensor(buf151, (672, 13), (13, 1), 0); del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_31.run(clone_11, buf157, buf159, 8736, 121, grid=grid(8736), stream=stream0)
        buf160 = buf152; del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_32.run(buf159, buf160, 672, 13, grid=grid(672), stream=stream0)
        buf161 = reinterpret_tensor(buf159, (672, 13), (1, 672), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_33.run(clone_11, buf157, convolution_41, unsqueeze_330, buf161, 8736, 121, grid=grid(8736), stream=stream0)
        buf162 = empty((672, ), device='cuda', dtype=torch.float32)
        buf163 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_34.run(buf161, squeeze_100, buf162, buf163, 672, 13, grid=grid(672), stream=stream0)
        del buf161
        buf164 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_35.run(clone_11, buf157, convolution_41, unsqueeze_330, buf162, squeeze_100, buf160, primals_67, buf164, 1568, 672, grid=grid(1568, 672), stream=stream0)
        del buf157
        del buf162
        del clone_11
        del convolution_41
        del primals_67
        del squeeze_100
        del unsqueeze_330
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf165 = aten.convolution_backward(buf164, add_186, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_186
        del buf164
        del primals_144
        buf166 = buf165[0]
        buf167 = buf165[1]
        del buf165
        buf168 = buf131; del buf131  # reuse
        buf169 = empty((112, ), device='cuda', dtype=torch.float32)
        buf170 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_45.run(buf127, buf166, convolution_40, unsqueeze_342, squeeze_97, buf168, buf169, buf170, 112, 1568, grid=grid(112), stream=stream0)
        buf171 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_46.run(buf171, buf166, convolution_40, unsqueeze_342, buf169, squeeze_97, buf168, primals_65, 896, 196, grid=grid(896, 196), stream=stream0)
        del buf166
        del buf169
        del convolution_40
        del primals_65
        del squeeze_97
        del unsqueeze_342
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf172 = aten.convolution_backward(buf171, mul_238, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf171
        del mul_238
        del primals_143
        buf173 = buf172[0]
        buf174 = buf172[1]
        del buf172
        buf175 = reinterpret_tensor(buf71, (8, 480, 1, 1, 2), (960, 2, 7680, 7680, 1), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_47.run(buf173, div_13, buf175, 7680, 98, grid=grid(7680), stream=stream0)
        del div_13
        buf176 = reinterpret_tensor(buf85, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf85  # reuse
        buf177 = reinterpret_tensor(buf176, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_48.run(buf177, buf175, bitwise_and_4, 3840, 2, grid=grid(3840), stream=stream0)
        del bitwise_and_4
        del buf175
        buf178 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_49.run(buf177, buf178, 480, 8, grid=grid(480), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf179 = aten.convolution_backward(buf177, relu_14, primals_141, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf177
        del primals_141
        buf180 = buf179[0]
        buf181 = buf179[1]
        del buf179
        buf182 = buf180; del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]
        triton_poi_fused_hardswish_backward_threshold_backward_50.run(buf182, relu_14, 960, grid=grid(960), stream=stream0)
        del relu_14
        buf183 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_51.run(buf182, buf183, 120, 8, grid=grid(120), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf184 = aten.convolution_backward(buf182, mean_3, primals_139, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_3
        del primals_139
        buf185 = buf184[0]
        buf186 = buf184[1]
        del buf184
        buf187 = empty((480, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_52.run(clone_10, buf173, div_14, buf185, buf187, 6240, 121, grid=grid(6240), stream=stream0)
        buf188 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_53.run(buf187, buf188, 480, 13, grid=grid(480), stream=stream0)
        buf189 = reinterpret_tensor(buf187, (480, 13), (1, 480), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_54.run(clone_10, buf173, div_14, buf185, convolution_37, unsqueeze_354, buf189, 6240, 121, grid=grid(6240), stream=stream0)
        buf190 = empty((480, ), device='cuda', dtype=torch.float32)
        buf192 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_55.run(buf189, squeeze_94, buf190, buf192, 480, 13, grid=grid(480), stream=stream0)
        buf191 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        buf193 = buf191; del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_56.run(buf193, clone_10, buf173, div_14, buf185, convolution_37, unsqueeze_354, buf190, squeeze_94, buf188, primals_63, 1568, 480, grid=grid(1568, 480), stream=stream0)
        del buf173
        del buf185
        del clone_10
        del convolution_37
        del div_14
        del primals_63
        del squeeze_94
        del unsqueeze_354
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf194 = aten.convolution_backward(buf193, div_12, primals_138, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
        del div_12
        del primals_138
        buf195 = buf194[0]
        buf196 = buf194[1]
        del buf194
        buf197 = reinterpret_tensor(buf189, (480, 13), (13, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_57.run(clone_9, buf195, buf197, 6240, 121, grid=grid(6240), stream=stream0)
        buf198 = buf190; del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_53.run(buf197, buf198, 480, 13, grid=grid(480), stream=stream0)
        buf199 = reinterpret_tensor(buf197, (480, 13), (1, 480), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_58.run(clone_9, buf195, convolution_36, unsqueeze_366, buf199, 6240, 121, grid=grid(6240), stream=stream0)
        buf200 = empty((480, ), device='cuda', dtype=torch.float32)
        buf201 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_55.run(buf199, squeeze_91, buf200, buf201, 480, 13, grid=grid(480), stream=stream0)
        del buf199
        buf202 = buf193; del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_59.run(clone_9, buf195, convolution_36, unsqueeze_366, buf200, squeeze_91, buf198, primals_61, buf202, 1568, 480, grid=grid(1568, 480), stream=stream0)
        del buf195
        del buf200
        del clone_9
        del convolution_36
        del primals_61
        del squeeze_91
        del unsqueeze_366
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf203 = aten.convolution_backward(buf202, add_168, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_168
        del primals_137
        buf204 = buf203[0]
        buf205 = buf203[1]
        del buf203
        buf206 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_60.run(buf204, buf206, 80, 1568, grid=grid(80), stream=stream0)
        buf207 = empty((80, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_61.run(buf204, convolution_35, unsqueeze_378, buf207, 1040, 121, grid=grid(1040), stream=stream0)
        buf208 = empty((80, ), device='cuda', dtype=torch.float32)
        buf209 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_62.run(buf207, squeeze_88, buf208, buf209, 80, 13, grid=grid(80), stream=stream0)
        del buf207
        buf210 = empty((8, 80, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_63.run(buf204, convolution_35, unsqueeze_378, buf208, squeeze_88, buf206, primals_59, buf210, 640, 196, grid=grid(640, 196), stream=stream0)
        del convolution_35
        del primals_59
        del squeeze_88
        del unsqueeze_378
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf211 = aten.convolution_backward(buf210, div_11, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_11
        del primals_136
        buf212 = buf211[0]
        buf213 = buf211[1]
        del buf211
        buf214 = empty((184, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_64.run(clone_8, buf212, buf214, 2392, 121, grid=grid(2392), stream=stream0)
        buf215 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_65.run(buf214, buf215, 184, 13, grid=grid(184), stream=stream0)
        buf216 = reinterpret_tensor(buf214, (184, 13), (1, 184), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_66.run(clone_8, buf212, convolution_34, unsqueeze_390, buf216, 2392, 121, grid=grid(2392), stream=stream0)
        buf217 = empty((184, ), device='cuda', dtype=torch.float32)
        buf218 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_67.run(buf216, squeeze_85, buf217, buf218, 184, 13, grid=grid(184), stream=stream0)
        buf219 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_68.run(clone_8, buf212, convolution_34, unsqueeze_390, buf217, squeeze_85, buf215, primals_57, buf219, 1568, 184, grid=grid(1568, 184), stream=stream0)
        del buf212
        del clone_8
        del convolution_34
        del primals_57
        del squeeze_85
        del unsqueeze_390
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf220 = aten.convolution_backward(buf219, div_10, primals_135, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 184, [True, True, False])
        del div_10
        del primals_135
        buf221 = buf220[0]
        buf222 = buf220[1]
        del buf220
        buf223 = reinterpret_tensor(buf216, (184, 13), (13, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_64.run(clone_7, buf221, buf223, 2392, 121, grid=grid(2392), stream=stream0)
        buf224 = buf217; del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_65.run(buf223, buf224, 184, 13, grid=grid(184), stream=stream0)
        buf225 = reinterpret_tensor(buf223, (184, 13), (1, 184), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_66.run(clone_7, buf221, convolution_33, unsqueeze_402, buf225, 2392, 121, grid=grid(2392), stream=stream0)
        buf226 = empty((184, ), device='cuda', dtype=torch.float32)
        buf227 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_67.run(buf225, squeeze_82, buf226, buf227, 184, 13, grid=grid(184), stream=stream0)
        buf228 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_68.run(clone_7, buf221, convolution_33, unsqueeze_402, buf226, squeeze_82, buf224, primals_55, buf228, 1568, 184, grid=grid(1568, 184), stream=stream0)
        del buf221
        del clone_7
        del convolution_33
        del primals_55
        del squeeze_82
        del unsqueeze_402
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf229 = aten.convolution_backward(buf228, add_150, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_150
        del primals_134
        buf230 = buf229[0]
        buf231 = buf229[1]
        del buf229
        buf232 = buf208; del buf208  # reuse
        buf233 = empty((80, ), device='cuda', dtype=torch.float32)
        buf234 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_69.run(buf204, buf230, convolution_32, unsqueeze_414, squeeze_79, buf232, buf233, buf234, 80, 1568, grid=grid(80), stream=stream0)
        buf235 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_70.run(buf204, buf230, convolution_32, unsqueeze_414, buf233, squeeze_79, buf232, primals_53, buf235, 640, 196, grid=grid(640, 196), stream=stream0)
        del convolution_32
        del primals_53
        del squeeze_79
        del unsqueeze_414
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf236 = aten.convolution_backward(buf235, div_9, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_9
        del primals_133
        buf237 = buf236[0]
        buf238 = buf236[1]
        del buf236
        buf239 = reinterpret_tensor(buf225, (184, 13), (13, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_64.run(clone_6, buf237, buf239, 2392, 121, grid=grid(2392), stream=stream0)
        buf240 = buf226; del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_65.run(buf239, buf240, 184, 13, grid=grid(184), stream=stream0)
        buf241 = reinterpret_tensor(buf239, (184, 13), (1, 184), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_66.run(clone_6, buf237, convolution_31, unsqueeze_426, buf241, 2392, 121, grid=grid(2392), stream=stream0)
        buf242 = empty((184, ), device='cuda', dtype=torch.float32)
        buf243 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_67.run(buf241, squeeze_76, buf242, buf243, 184, 13, grid=grid(184), stream=stream0)
        buf244 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_68.run(clone_6, buf237, convolution_31, unsqueeze_426, buf242, squeeze_76, buf240, primals_51, buf244, 1568, 184, grid=grid(1568, 184), stream=stream0)
        del buf237
        del clone_6
        del convolution_31
        del primals_51
        del squeeze_76
        del unsqueeze_426
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf245 = aten.convolution_backward(buf244, div_8, primals_132, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 184, [True, True, False])
        del div_8
        del primals_132
        buf246 = buf245[0]
        buf247 = buf245[1]
        del buf245
        buf248 = reinterpret_tensor(buf241, (184, 13), (13, 1), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_64.run(clone_5, buf246, buf248, 2392, 121, grid=grid(2392), stream=stream0)
        buf249 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_65.run(buf248, buf249, 184, 13, grid=grid(184), stream=stream0)
        buf250 = reinterpret_tensor(buf248, (184, 13), (1, 184), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_66.run(clone_5, buf246, convolution_30, unsqueeze_438, buf250, 2392, 121, grid=grid(2392), stream=stream0)
        buf251 = empty((184, ), device='cuda', dtype=torch.float32)
        buf252 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_67.run(buf250, squeeze_73, buf251, buf252, 184, 13, grid=grid(184), stream=stream0)
        del buf250
        buf253 = buf244; del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_68.run(clone_5, buf246, convolution_30, unsqueeze_438, buf251, squeeze_73, buf249, primals_49, buf253, 1568, 184, grid=grid(1568, 184), stream=stream0)
        del buf246
        del buf251
        del clone_5
        del convolution_30
        del primals_49
        del squeeze_73
        del unsqueeze_438
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf254 = aten.convolution_backward(buf253, add_132, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_132
        del buf253
        del primals_131
        buf255 = buf254[0]
        buf256 = buf254[1]
        del buf254
        buf257 = buf233; del buf233  # reuse
        buf258 = empty((80, ), device='cuda', dtype=torch.float32)
        buf260 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_71.run(buf204, buf230, buf255, convolution_29, unsqueeze_450, squeeze_70, buf257, buf258, buf260, 80, 1568, grid=grid(80), stream=stream0)
        buf259 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_72.run(buf204, buf230, buf255, convolution_29, unsqueeze_450, buf258, squeeze_70, buf257, primals_47, buf259, 640, 196, grid=grid(640, 196), stream=stream0)
        del convolution_29
        del primals_47
        del squeeze_70
        del unsqueeze_450
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf261 = aten.convolution_backward(buf259, div_7, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf259
        del div_7
        del primals_130
        buf262 = buf261[0]
        buf263 = buf261[1]
        del buf261
        buf264 = empty((200, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_73.run(clone_4, buf262, buf264, 2600, 121, grid=grid(2600), stream=stream0)
        buf265 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_74.run(buf264, buf265, 200, 13, grid=grid(200), stream=stream0)
        buf266 = reinterpret_tensor(buf264, (200, 13), (1, 200), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_75.run(clone_4, buf262, convolution_28, unsqueeze_462, buf266, 2600, 121, grid=grid(2600), stream=stream0)
        buf267 = empty((200, ), device='cuda', dtype=torch.float32)
        buf268 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_76.run(buf266, squeeze_67, buf267, buf268, 200, 13, grid=grid(200), stream=stream0)
        buf269 = empty_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_77.run(clone_4, buf262, convolution_28, unsqueeze_462, buf267, squeeze_67, buf265, primals_45, buf269, 1568, 200, grid=grid(1568, 200), stream=stream0)
        del buf262
        del clone_4
        del convolution_28
        del primals_45
        del squeeze_67
        del unsqueeze_462
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf270 = aten.convolution_backward(buf269, div_6, primals_129, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 200, [True, True, False])
        del div_6
        del primals_129
        buf271 = buf270[0]
        buf272 = buf270[1]
        del buf270
        buf273 = reinterpret_tensor(buf266, (200, 13), (13, 1), 0); del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_73.run(clone_3, buf271, buf273, 2600, 121, grid=grid(2600), stream=stream0)
        buf274 = buf267; del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_74.run(buf273, buf274, 200, 13, grid=grid(200), stream=stream0)
        buf275 = reinterpret_tensor(buf273, (200, 13), (1, 200), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_75.run(clone_3, buf271, convolution_27, unsqueeze_474, buf275, 2600, 121, grid=grid(2600), stream=stream0)
        buf276 = empty((200, ), device='cuda', dtype=torch.float32)
        buf277 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_76.run(buf275, squeeze_64, buf276, buf277, 200, 13, grid=grid(200), stream=stream0)
        del buf275
        buf278 = buf269; del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_77.run(clone_3, buf271, convolution_27, unsqueeze_474, buf276, squeeze_64, buf274, primals_43, buf278, 1568, 200, grid=grid(1568, 200), stream=stream0)
        del buf271
        del buf276
        del clone_3
        del convolution_27
        del primals_43
        del squeeze_64
        del unsqueeze_474
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf279 = aten.convolution_backward(buf278, add_114, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_114
        del buf278
        del primals_128
        buf280 = buf279[0]
        buf281 = buf279[1]
        del buf279
        buf282 = buf258; del buf258  # reuse
        buf283 = empty((80, ), device='cuda', dtype=torch.float32)
        buf285 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_78.run(buf204, buf230, buf255, buf280, convolution_26, unsqueeze_486, squeeze_61, buf282, buf283, buf285, 80, 1568, grid=grid(80), stream=stream0)
        buf284 = buf204; del buf204  # reuse
        buf286 = buf284; del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_79.run(buf286, buf230, buf255, buf280, convolution_26, unsqueeze_486, buf283, squeeze_61, buf282, primals_41, 640, 196, grid=grid(640, 196), stream=stream0)
        del buf230
        del buf255
        del buf280
        del buf283
        del convolution_26
        del primals_41
        del squeeze_61
        del unsqueeze_486
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf287 = aten.convolution_backward(buf286, div_5, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf286
        del div_5
        del primals_127
        buf288 = buf287[0]
        buf289 = buf287[1]
        del buf287
        buf290 = empty((240, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_80.run(clone_2, buf288, buf290, 3120, 121, grid=grid(3120), stream=stream0)
        buf291 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_81.run(buf290, buf291, 240, 13, grid=grid(240), stream=stream0)
        buf292 = reinterpret_tensor(buf290, (240, 13), (1, 240), 0); del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_82.run(clone_2, buf288, convolution_25, unsqueeze_498, buf292, 3120, 121, grid=grid(3120), stream=stream0)
        buf293 = empty((240, ), device='cuda', dtype=torch.float32)
        buf294 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_83.run(buf292, squeeze_58, buf293, buf294, 240, 13, grid=grid(240), stream=stream0)
        del buf292
        buf295 = reinterpret_tensor(buf88, (8, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_84.run(clone_2, buf288, convolution_25, unsqueeze_498, buf293, squeeze_58, buf291, primals_39, buf295, 1568, 240, grid=grid(1568, 240), stream=stream0)
        del buf288
        del clone_2
        del convolution_25
        del primals_39
        del squeeze_58
        del unsqueeze_498
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf296 = aten.convolution_backward(buf295, div_4, primals_126, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False])
        del buf295
        del div_4
        del primals_126
        buf297 = buf296[0]
        buf298 = buf296[1]
        del buf296
        buf299 = empty((240, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_85.run(clone_1, buf297, buf299, 11760, 128, grid=grid(11760), stream=stream0)
        buf300 = buf293; del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_86.run(buf299, buf300, 240, 49, grid=grid(240), stream=stream0)
        buf301 = reinterpret_tensor(buf299, (240, 49), (1, 240), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_87.run(clone_1, buf297, convolution_24, unsqueeze_510, buf301, 11760, 128, grid=grid(11760), stream=stream0)
        buf302 = empty((240, ), device='cuda', dtype=torch.float32)
        buf303 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_88.run(buf301, squeeze_55, buf302, buf303, 240, 49, grid=grid(240), stream=stream0)
        del buf301
        buf304 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_89.run(clone_1, buf297, convolution_24, unsqueeze_510, buf302, squeeze_55, buf300, primals_37, buf304, 6272, 240, grid=grid(6272, 240), stream=stream0)
        del buf297
        del buf302
        del clone_1
        del convolution_24
        del primals_37
        del squeeze_55
        del unsqueeze_510
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf305 = aten.convolution_backward(buf304, add_97, primals_125, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_97
        del buf304
        del primals_125
        buf306 = buf305[0]
        buf307 = buf305[1]
        del buf305
        buf308 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_90.run(buf306, buf308, 40, 6272, grid=grid(40), stream=stream0)
        buf309 = empty((40, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_91.run(buf306, convolution_23, unsqueeze_522, buf309, 1960, 128, grid=grid(1960), stream=stream0)
        buf310 = empty((40, ), device='cuda', dtype=torch.float32)
        buf311 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_92.run(buf309, squeeze_52, buf310, buf311, 40, 49, grid=grid(40), stream=stream0)
        del buf309
        buf312 = empty((8, 40, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_93.run(buf306, convolution_23, unsqueeze_522, buf310, squeeze_52, buf308, primals_35, buf312, 320, 784, grid=grid(320, 784), stream=stream0)
        del convolution_23
        del primals_35
        del squeeze_52
        del unsqueeze_522
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf313 = aten.convolution_backward(buf312, mul_122, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_122
        del primals_124
        buf314 = buf313[0]
        buf315 = buf313[1]
        del buf313
        buf316 = empty_strided((8, 120, 1, 1, 7), (840, 7, 6720, 6720, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_94.run(buf314, relu_12, buf316, 6720, 112, grid=grid(6720), stream=stream0)
        buf317 = reinterpret_tensor(buf182, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf182  # reuse
        buf318 = reinterpret_tensor(buf317, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_95.run(buf318, buf316, bitwise_and_5, 960, 7, grid=grid(960), stream=stream0)
        del bitwise_and_5
        buf319 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_51.run(buf318, buf319, 120, 8, grid=grid(120), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf320 = aten.convolution_backward(buf318, relu_13, primals_122, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf318
        del primals_122
        buf321 = buf320[0]
        buf322 = buf320[1]
        del buf320
        buf323 = buf321; del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]
        triton_poi_fused_hardswish_backward_threshold_backward_96.run(buf323, relu_13, 256, grid=grid(256), stream=stream0)
        del relu_13
        buf324 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_97.run(buf323, buf324, 32, 8, grid=grid(32), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf325 = aten.convolution_backward(buf323, mean_2, primals_120, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf323
        del mean_2
        del primals_120
        buf326 = buf325[0]
        buf327 = buf325[1]
        del buf325
        buf328 = empty((120, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_98.run(relu_12, buf314, div_3, buf326, buf328, 5880, 128, grid=grid(5880), stream=stream0)
        buf329 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_99.run(buf328, buf329, 120, 49, grid=grid(120), stream=stream0)
        buf330 = reinterpret_tensor(buf328, (120, 49), (1, 120), 0); del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_100.run(relu_12, buf314, div_3, buf326, convolution_20, unsqueeze_534, buf330, 5880, 128, grid=grid(5880), stream=stream0)
        buf331 = empty((120, ), device='cuda', dtype=torch.float32)
        buf333 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_101.run(buf330, squeeze_49, buf331, buf333, 120, 49, grid=grid(120), stream=stream0)
        buf332 = reinterpret_tensor(buf202, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf202  # reuse
        buf334 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_102.run(buf334, relu_12, buf314, div_3, buf326, convolution_20, unsqueeze_534, buf331, squeeze_49, buf329, primals_33, 6272, 120, grid=grid(6272, 120), stream=stream0)
        del buf314
        del convolution_20
        del div_3
        del primals_33
        del relu_12
        del squeeze_49
        del unsqueeze_534
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf335 = aten.convolution_backward(buf334, relu_11, primals_119, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del primals_119
        buf336 = buf335[0]
        buf337 = buf335[1]
        del buf335
        buf338 = reinterpret_tensor(buf330, (120, 49), (49, 1), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_103.run(relu_11, buf336, buf338, 5880, 128, grid=grid(5880), stream=stream0)
        buf339 = buf331; del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_99.run(buf338, buf339, 120, 49, grid=grid(120), stream=stream0)
        buf340 = reinterpret_tensor(buf338, (120, 49), (1, 120), 0); del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_104.run(relu_11, buf336, convolution_19, unsqueeze_546, buf340, 5880, 128, grid=grid(5880), stream=stream0)
        buf341 = empty((120, ), device='cuda', dtype=torch.float32)
        buf342 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_101.run(buf340, squeeze_46, buf341, buf342, 120, 49, grid=grid(120), stream=stream0)
        buf343 = buf334; del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_105.run(relu_11, buf336, convolution_19, unsqueeze_546, buf341, squeeze_46, buf339, primals_31, buf343, 6272, 120, grid=grid(6272, 120), stream=stream0)
        del buf336
        del convolution_19
        del primals_31
        del relu_11
        del squeeze_46
        del unsqueeze_546
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf344 = aten.convolution_backward(buf343, add_80, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_80
        del primals_118
        buf345 = buf344[0]
        buf346 = buf344[1]
        del buf344
        buf347 = buf310; del buf310  # reuse
        buf348 = empty((40, ), device='cuda', dtype=torch.float32)
        buf349 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_106.run(buf306, buf345, convolution_18, unsqueeze_558, squeeze_43, buf347, buf348, buf349, 40, 6272, grid=grid(40), stream=stream0)
        buf350 = buf312; del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_107.run(buf306, buf345, convolution_18, unsqueeze_558, buf348, squeeze_43, buf347, primals_29, buf350, 320, 784, grid=grid(320, 784), stream=stream0)
        del convolution_18
        del primals_29
        del squeeze_43
        del unsqueeze_558
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf351 = aten.convolution_backward(buf350, mul_100, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf350
        del mul_100
        del primals_117
        buf352 = buf351[0]
        buf353 = buf351[1]
        del buf351
        buf354 = buf316; del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_94.run(buf352, relu_9, buf354, 6720, 112, grid=grid(6720), stream=stream0)
        buf355 = reinterpret_tensor(buf326, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf326  # reuse
        buf356 = reinterpret_tensor(buf355, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_95.run(buf356, buf354, bitwise_and_6, 960, 7, grid=grid(960), stream=stream0)
        del bitwise_and_6
        del buf354
        buf357 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_51.run(buf356, buf357, 120, 8, grid=grid(120), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf358 = aten.convolution_backward(buf356, relu_10, primals_115, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf356
        del primals_115
        buf359 = buf358[0]
        buf360 = buf358[1]
        del buf358
        buf361 = buf359; del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]
        triton_poi_fused_hardswish_backward_threshold_backward_96.run(buf361, relu_10, 256, grid=grid(256), stream=stream0)
        del relu_10
        buf362 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_97.run(buf361, buf362, 32, 8, grid=grid(32), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf363 = aten.convolution_backward(buf361, mean_1, primals_113, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf361
        del mean_1
        del primals_113
        buf364 = buf363[0]
        buf365 = buf363[1]
        del buf363
        buf366 = reinterpret_tensor(buf340, (120, 49), (49, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_98.run(relu_9, buf352, div_2, buf364, buf366, 5880, 128, grid=grid(5880), stream=stream0)
        buf367 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_99.run(buf366, buf367, 120, 49, grid=grid(120), stream=stream0)
        buf368 = reinterpret_tensor(buf366, (120, 49), (1, 120), 0); del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_100.run(relu_9, buf352, div_2, buf364, convolution_15, unsqueeze_570, buf368, 5880, 128, grid=grid(5880), stream=stream0)
        buf369 = empty((120, ), device='cuda', dtype=torch.float32)
        buf371 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_101.run(buf368, squeeze_40, buf369, buf371, 120, 49, grid=grid(120), stream=stream0)
        buf370 = buf343; del buf343  # reuse
        buf372 = buf370; del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_102.run(buf372, relu_9, buf352, div_2, buf364, convolution_15, unsqueeze_570, buf369, squeeze_40, buf367, primals_27, 6272, 120, grid=grid(6272, 120), stream=stream0)
        del buf352
        del buf364
        del convolution_15
        del div_2
        del primals_27
        del relu_9
        del squeeze_40
        del unsqueeze_570
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf373 = aten.convolution_backward(buf372, relu_8, primals_112, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del primals_112
        buf374 = buf373[0]
        buf375 = buf373[1]
        del buf373
        buf376 = reinterpret_tensor(buf368, (120, 49), (49, 1), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_103.run(relu_8, buf374, buf376, 5880, 128, grid=grid(5880), stream=stream0)
        buf377 = buf369; del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_99.run(buf376, buf377, 120, 49, grid=grid(120), stream=stream0)
        buf378 = reinterpret_tensor(buf376, (120, 49), (1, 120), 0); del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_104.run(relu_8, buf374, convolution_14, unsqueeze_582, buf378, 5880, 128, grid=grid(5880), stream=stream0)
        buf379 = empty((120, ), device='cuda', dtype=torch.float32)
        buf380 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_101.run(buf378, squeeze_37, buf379, buf380, 120, 49, grid=grid(120), stream=stream0)
        del buf378
        buf381 = buf372; del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_105.run(relu_8, buf374, convolution_14, unsqueeze_582, buf379, squeeze_37, buf377, primals_25, buf381, 6272, 120, grid=grid(6272, 120), stream=stream0)
        del buf374
        del buf379
        del convolution_14
        del primals_25
        del relu_8
        del squeeze_37
        del unsqueeze_582
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf382 = aten.convolution_backward(buf381, add_63, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_63
        del buf381
        del primals_111
        buf383 = buf382[0]
        buf384 = buf382[1]
        del buf382
        buf385 = buf348; del buf348  # reuse
        buf386 = empty((40, ), device='cuda', dtype=torch.float32)
        buf388 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_108.run(buf306, buf345, buf383, convolution_13, unsqueeze_594, squeeze_34, buf385, buf386, buf388, 40, 6272, grid=grid(40), stream=stream0)
        buf387 = buf306; del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_109.run(buf387, buf345, buf383, convolution_13, unsqueeze_594, buf386, squeeze_34, buf385, primals_23, 320, 784, grid=grid(320, 784), stream=stream0)
        del buf345
        del buf383
        del buf386
        del convolution_13
        del primals_23
        del squeeze_34
        del unsqueeze_594
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf389 = aten.convolution_backward(buf387, mul_78, primals_110, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf387
        del mul_78
        del primals_110
        buf390 = buf389[0]
        buf391 = buf389[1]
        del buf389
        buf392 = empty_strided((8, 72, 1, 1, 7), (504, 7, 4032, 4032, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_110.run(buf390, relu_6, buf392, 4032, 112, grid=grid(4032), stream=stream0)
        buf393 = empty_strided((8, 72, 1, 1), (72, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf394 = reinterpret_tensor(buf393, (8, 72, 1, 1), (72, 1, 72, 72), 0); del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_111.run(buf394, buf392, bitwise_and_7, 576, 7, grid=grid(576), stream=stream0)
        del bitwise_and_7
        del buf392
        buf395 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_112.run(buf394, buf395, 72, 8, grid=grid(72), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf396 = aten.convolution_backward(buf394, relu_7, primals_108, [72], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf394
        del primals_108
        buf397 = buf396[0]
        buf398 = buf396[1]
        del buf396
        buf399 = buf397; del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]
        triton_poi_fused_hardswish_backward_threshold_backward_113.run(buf399, relu_7, 192, grid=grid(192), stream=stream0)
        del relu_7
        buf400 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_114.run(buf399, buf400, 24, 8, grid=grid(24), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf401 = aten.convolution_backward(buf399, mean, primals_106, [24], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf399
        del mean
        del primals_106
        buf402 = buf401[0]
        buf403 = buf401[1]
        del buf401
        buf404 = empty((72, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_115.run(relu_6, buf390, div_1, buf402, buf404, 3528, 128, grid=grid(3528), stream=stream0)
        buf405 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_116.run(buf404, buf405, 72, 49, grid=grid(72), stream=stream0)
        buf406 = reinterpret_tensor(buf404, (72, 49), (1, 72), 0); del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_117.run(relu_6, buf390, div_1, buf402, convolution_10, unsqueeze_606, buf406, 3528, 128, grid=grid(3528), stream=stream0)
        buf407 = empty((72, ), device='cuda', dtype=torch.float32)
        buf409 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_118.run(buf406, squeeze_31, buf407, buf409, 72, 49, grid=grid(72), stream=stream0)
        del buf406
        buf408 = empty_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cuda', dtype=torch.float32)
        buf410 = buf408; del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_119.run(buf410, relu_6, buf390, div_1, buf402, convolution_10, unsqueeze_606, buf407, squeeze_31, buf405, primals_21, 6272, 72, grid=grid(6272, 72), stream=stream0)
        del buf390
        del buf402
        del convolution_10
        del div_1
        del primals_21
        del relu_6
        del squeeze_31
        del unsqueeze_606
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf411 = aten.convolution_backward(buf410, relu_5, primals_105, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 72, [True, True, False])
        del buf410
        del primals_105
        buf412 = buf411[0]
        buf413 = buf411[1]
        del buf411
        buf414 = empty((72, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_120.run(relu_5, buf412, buf414, 14112, 128, grid=grid(14112), stream=stream0)
        buf415 = buf407; del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_121.run(buf414, buf415, 72, 196, grid=grid(72), stream=stream0)
        buf416 = reinterpret_tensor(buf414, (72, 196), (1, 72), 0); del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_122.run(relu_5, buf412, convolution_9, unsqueeze_618, buf416, 14112, 128, grid=grid(14112), stream=stream0)
        buf417 = empty((72, ), device='cuda', dtype=torch.float32)
        buf418 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_123.run(buf416, squeeze_28, buf417, buf418, 72, 196, grid=grid(72), stream=stream0)
        buf419 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_124.run(relu_5, buf412, convolution_9, unsqueeze_618, buf417, squeeze_28, buf415, primals_19, buf419, 25088, 72, grid=grid(25088, 72), stream=stream0)
        del buf412
        del convolution_9
        del primals_19
        del relu_5
        del squeeze_28
        del unsqueeze_618
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf420 = aten.convolution_backward(buf419, add_47, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_47
        del primals_104
        buf421 = buf420[0]
        buf422 = buf420[1]
        del buf420
        buf423 = empty_strided((24, 4), (1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_125.run(buf421, buf423, 96, 6272, grid=grid(96), stream=stream0)
        buf424 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_126.run(buf423, buf424, 24, 4, grid=grid(24), stream=stream0)
        buf425 = empty((24, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_127.run(buf421, convolution_8, unsqueeze_630, buf425, 4704, 128, grid=grid(4704), stream=stream0)
        buf426 = empty((24, ), device='cuda', dtype=torch.float32)
        buf427 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_128.run(buf425, squeeze_25, buf426, buf427, 24, 196, grid=grid(24), stream=stream0)
        del buf425
        buf428 = empty((8, 24, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_129.run(buf421, convolution_8, unsqueeze_630, buf426, squeeze_25, buf424, primals_17, buf428, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del convolution_8
        del primals_17
        del squeeze_25
        del unsqueeze_630
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf429 = aten.convolution_backward(buf428, relu_4, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf428
        del primals_103
        buf430 = buf429[0]
        buf431 = buf429[1]
        del buf429
        buf432 = reinterpret_tensor(buf416, (72, 196), (196, 1), 0); del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_120.run(relu_4, buf430, buf432, 14112, 128, grid=grid(14112), stream=stream0)
        buf433 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_121.run(buf432, buf433, 72, 196, grid=grid(72), stream=stream0)
        buf434 = reinterpret_tensor(buf432, (72, 196), (1, 72), 0); del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_122.run(relu_4, buf430, convolution_7, unsqueeze_642, buf434, 14112, 128, grid=grid(14112), stream=stream0)
        buf435 = empty((72, ), device='cuda', dtype=torch.float32)
        buf436 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_123.run(buf434, squeeze_22, buf435, buf436, 72, 196, grid=grid(72), stream=stream0)
        buf437 = buf419; del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_124.run(relu_4, buf430, convolution_7, unsqueeze_642, buf435, squeeze_22, buf433, primals_15, buf437, 25088, 72, grid=grid(25088, 72), stream=stream0)
        del buf430
        del convolution_7
        del primals_15
        del relu_4
        del squeeze_22
        del unsqueeze_642
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf438 = aten.convolution_backward(buf437, relu_3, primals_102, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False])
        del primals_102
        buf439 = buf438[0]
        buf440 = buf438[1]
        del buf438
        buf441 = reinterpret_tensor(buf434, (72, 196), (196, 1), 0); del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_120.run(relu_3, buf439, buf441, 14112, 128, grid=grid(14112), stream=stream0)
        buf442 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_121.run(buf441, buf442, 72, 196, grid=grid(72), stream=stream0)
        buf443 = reinterpret_tensor(buf441, (72, 196), (1, 72), 0); del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_122.run(relu_3, buf439, convolution_6, unsqueeze_654, buf443, 14112, 128, grid=grid(14112), stream=stream0)
        buf444 = empty((72, ), device='cuda', dtype=torch.float32)
        buf445 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_123.run(buf443, squeeze_19, buf444, buf445, 72, 196, grid=grid(72), stream=stream0)
        del buf443
        buf446 = buf437; del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_124.run(relu_3, buf439, convolution_6, unsqueeze_654, buf444, squeeze_19, buf442, primals_13, buf446, 25088, 72, grid=grid(25088, 72), stream=stream0)
        del buf439
        del buf444
        del convolution_6
        del primals_13
        del relu_3
        del squeeze_19
        del unsqueeze_654
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf447 = aten.convolution_backward(buf446, add_31, primals_101, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_31
        del buf446
        del primals_101
        buf448 = buf447[0]
        buf449 = buf447[1]
        del buf447
        buf450 = buf423; del buf423  # reuse
        buf452 = empty_strided((24, 4), (1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_130.run(buf421, buf448, convolution_5, unsqueeze_666, buf450, buf452, 96, 6272, grid=grid(96), stream=stream0)
        buf451 = buf426; del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_126.run(buf450, buf451, 24, 4, grid=grid(24), stream=stream0)
        del buf450
        buf453 = empty((24, ), device='cuda', dtype=torch.float32)
        buf454 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_131.run(buf452, squeeze_16, buf453, buf454, 24, 4, grid=grid(24), stream=stream0)
        del buf452
        buf455 = buf421; del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_132.run(buf455, buf448, convolution_5, unsqueeze_666, buf453, squeeze_16, buf451, primals_11, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del buf448
        del buf453
        del convolution_5
        del primals_11
        del squeeze_16
        del unsqueeze_666
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf456 = aten.convolution_backward(buf455, relu_2, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf455
        del primals_100
        buf457 = buf456[0]
        buf458 = buf456[1]
        del buf456
        buf459 = empty((64, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_133.run(relu_2, buf457, buf459, 12544, 128, grid=grid(12544), stream=stream0)
        buf460 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_134.run(buf459, buf460, 64, 196, grid=grid(64), stream=stream0)
        buf461 = reinterpret_tensor(buf459, (64, 196), (1, 64), 0); del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_135.run(relu_2, buf457, convolution_4, unsqueeze_678, buf461, 12544, 128, grid=grid(12544), stream=stream0)
        buf462 = empty((64, ), device='cuda', dtype=torch.float32)
        buf463 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_136.run(buf461, squeeze_13, buf462, buf463, 64, 196, grid=grid(64), stream=stream0)
        buf464 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_137.run(relu_2, buf457, convolution_4, unsqueeze_678, buf462, squeeze_13, buf460, primals_9, buf464, 25088, 64, grid=grid(25088, 64), stream=stream0)
        del buf457
        del convolution_4
        del primals_9
        del relu_2
        del squeeze_13
        del unsqueeze_678
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf465 = aten.convolution_backward(buf464, relu_1, primals_99, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
        del primals_99
        buf466 = buf465[0]
        buf467 = buf465[1]
        del buf465
        buf468 = empty((64, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_138.run(relu_1, buf466, buf468, 50176, 128, grid=grid(50176), stream=stream0)
        buf469 = buf462; del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_139.run(buf468, buf469, 64, 784, grid=grid(64), stream=stream0)
        buf470 = reinterpret_tensor(buf468, (64, 784), (1, 64), 0); del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_140.run(relu_1, buf466, convolution_3, unsqueeze_690, buf470, 50176, 128, grid=grid(50176), stream=stream0)
        buf471 = empty((64, ), device='cuda', dtype=torch.float32)
        buf472 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_141.run(buf470, squeeze_10, buf471, buf472, 64, 784, grid=grid(64), stream=stream0)
        del buf470
        buf473 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_142.run(relu_1, buf466, convolution_3, unsqueeze_690, buf471, squeeze_10, buf469, primals_7, buf473, 100352, 64, grid=grid(100352, 64), stream=stream0)
        del buf466
        del buf471
        del convolution_3
        del primals_7
        del relu_1
        del squeeze_10
        del unsqueeze_690
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf474 = aten.convolution_backward(buf473, add_16, primals_98, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_16
        del buf473
        del primals_98
        buf475 = buf474[0]
        buf476 = buf474[1]
        del buf474
        buf477 = empty((16, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_143.run(buf475, buf477, 208, 7720, grid=grid(208), stream=stream0)
        buf478 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_144.run(buf477, buf478, 16, 13, grid=grid(16), stream=stream0)
        buf479 = reinterpret_tensor(buf461, (16, 784), (784, 1), 0); del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_145.run(buf475, convolution_2, unsqueeze_702, buf479, 12544, 128, grid=grid(12544), stream=stream0)
        buf480 = empty((16, ), device='cuda', dtype=torch.float32)
        buf481 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_146.run(buf479, squeeze_7, buf480, buf481, 16, 784, grid=grid(16), stream=stream0)
        buf482 = reinterpret_tensor(buf464, (8, 16, 112, 112), (200704, 12544, 112, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_147.run(buf475, convolution_2, unsqueeze_702, buf480, squeeze_7, buf478, primals_5, buf482, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del convolution_2
        del primals_5
        del squeeze_7
        del unsqueeze_702
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf483 = aten.convolution_backward(buf482, relu, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_97
        buf484 = buf483[0]
        buf485 = buf483[1]
        del buf483
        buf486 = buf479; del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_148.run(relu, buf484, buf486, 12544, 128, grid=grid(12544), stream=stream0)
        buf487 = buf480; del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_149.run(buf486, buf487, 16, 784, grid=grid(16), stream=stream0)
        buf488 = reinterpret_tensor(buf486, (16, 784), (1, 16), 0); del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_150.run(relu, buf484, convolution_1, unsqueeze_714, buf488, 12544, 128, grid=grid(12544), stream=stream0)
        buf489 = empty((16, ), device='cuda', dtype=torch.float32)
        buf490 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_151.run(buf488, squeeze_4, buf489, buf490, 16, 784, grid=grid(16), stream=stream0)
        buf491 = reinterpret_tensor(buf482, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_152.run(relu, buf484, convolution_1, unsqueeze_714, buf489, squeeze_4, buf487, primals_3, buf491, 100352, 16, grid=grid(100352, 16), stream=stream0)
        del buf484
        del convolution_1
        del primals_3
        del relu
        del squeeze_4
        del unsqueeze_714
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf492 = aten.convolution_backward(buf491, div, primals_96, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False])
        del div
        del primals_96
        buf493 = buf492[0]
        buf494 = buf492[1]
        del buf492
        buf495 = buf477; del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_hardswish_backward_native_batch_norm_backward_153.run(clone, buf475, buf493, buf495, 208, 7720, grid=grid(208), stream=stream0)
        buf496 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_144.run(buf495, buf496, 16, 13, grid=grid(16), stream=stream0)
        del buf495
        buf497 = reinterpret_tensor(buf488, (16, 784), (784, 1), 0); del buf488  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_hardswish_backward_native_batch_norm_backward_154.run(clone, buf475, buf493, convolution, unsqueeze_726, buf497, 12544, 128, grid=grid(12544), stream=stream0)
        buf498 = empty((16, ), device='cuda', dtype=torch.float32)
        buf500 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_146.run(buf497, squeeze_1, buf498, buf500, 16, 784, grid=grid(16), stream=stream0)
        del buf497
        buf499 = buf491; del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_hardswish_backward_native_batch_norm_backward_155.run(clone, buf475, buf493, convolution, unsqueeze_726, buf498, squeeze_1, buf496, primals_1, buf499, 100352, 16, grid=grid(100352, 16), stream=stream0)
        del buf475
        del buf493
        del buf498
        del clone
        del convolution
        del primals_1
        del squeeze_1
        del unsqueeze_726
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf501 = aten.convolution_backward(buf499, primals_313, primals_95, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf499
        del primals_313
        del primals_95
        buf502 = buf501[1]
        return (buf500, buf496, buf490, buf487, buf481, buf478, buf472, buf469, buf463, buf460, buf454, buf451, buf445, buf442, buf436, buf433, buf427, buf424, buf418, buf415, buf409, buf405, buf388, buf385, buf380, buf377, buf371, buf367, buf349, buf347, buf342, buf339, buf333, buf329, buf311, buf308, buf303, buf300, buf294, buf291, buf285, buf282, buf277, buf274, buf268, buf265, buf260, buf257, buf252, buf249, buf243, buf240, buf234, buf232, buf227, buf224, buf218, buf215, buf209, buf206, buf201, buf198, buf192, buf188, buf170, buf168, buf163, buf160, buf154, buf150, buf132, buf129, buf124, buf121, buf115, buf111, buf95, buf92, buf87, buf84, buf78, buf74, buf57, buf55, buf50, buf47, buf41, buf37, buf20, buf17, buf12, buf9, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), buf502, buf494, buf485, buf476, buf467, buf458, buf449, buf440, buf431, buf422, buf413, buf403, buf400, buf398, buf395, buf391, buf384, buf375, buf365, buf362, buf360, buf357, buf353, buf346, buf337, buf327, buf324, buf322, buf319, buf315, buf307, buf298, buf289, buf281, buf272, buf263, buf256, buf247, buf238, buf231, buf222, buf213, buf205, buf196, buf186, buf183, buf181, buf178, buf174, buf167, buf158, buf148, buf145, buf143, buf140, buf136, buf128, buf119, buf109, buf106, buf104, buf101, buf98, buf91, buf82, buf72, buf69, buf67, buf64, buf61, buf54, buf45, buf35, buf32, buf30, buf27, buf24, buf16, buf7, buf4, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((40, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((200, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((200, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((80, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((672, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((160, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((1280, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    div = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_16 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_31 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_47 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((8, 72, 1, 1), (72, 1, 72, 72), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((8, 24, 1, 1), (24, 1, 24, 24), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((8, 72, 1, 1), (72, 1, 72, 72), device='cuda:0', dtype=torch.float32)
    mul_78 = rand_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_63 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    mean_1 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    mul_100 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_80 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    mean_2 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    mul_122 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_97 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_1 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_2 = rand_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cuda:0', dtype=torch.float32)
    div_5 = rand_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_114 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_3 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda:0', dtype=torch.float32)
    div_6 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_4 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_132 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_5 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_6 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_150 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_7 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_8 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_168 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_9 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_10 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    mean_3 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    mul_238 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_186 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_11 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_12 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    mean_4 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((8, 168, 1, 1), (168, 1, 168, 168), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    mul_262 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_205 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_13 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_14 = rand_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda:0', dtype=torch.float32)
    mean_5 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((8, 168, 1, 1), (168, 1, 168, 168), device='cuda:0', dtype=torch.float32)
    div_20 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    mul_286 = rand_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_223 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_15 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_16 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    mean_6 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cuda:0', dtype=torch.float32)
    mul_310 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_242 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_17 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_18 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    mean_7 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cuda:0', dtype=torch.float32)
    mul_334 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_261 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_19 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    mean_8 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((8, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((8, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_186 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_198 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cuda:0', dtype=torch.bool)
    unsqueeze_210 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_222 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_234 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_1 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cuda:0', dtype=torch.bool)
    unsqueeze_246 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_270 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_2 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.bool)
    unsqueeze_282 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_294 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_3 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.bool)
    unsqueeze_318 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_342 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_4 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.bool)
    unsqueeze_354 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_366 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_390 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_414 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_438 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_450 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_462 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_486 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_498 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_510 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_522 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_5 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.bool)
    unsqueeze_534 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_546 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_558 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_6 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.bool)
    unsqueeze_570 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_582 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_594 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_7 = rand_strided((8, 72, 1, 1), (72, 1, 72, 72), device='cuda:0', dtype=torch.bool)
    unsqueeze_606 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_618 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_630 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_642 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_666 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_678 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_690 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_702 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_714 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_726 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_108, primals_110, primals_111, primals_112, primals_113, primals_115, primals_117, primals_118, primals_119, primals_120, primals_122, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_141, primals_143, primals_144, primals_145, primals_146, primals_148, primals_150, primals_151, primals_152, primals_153, primals_155, primals_157, primals_158, primals_159, primals_160, primals_162, primals_164, primals_165, primals_166, primals_167, primals_169, primals_171, primals_172, primals_173, primals_313, convolution, squeeze_1, clone, div, convolution_1, squeeze_4, relu, convolution_2, squeeze_7, add_16, convolution_3, squeeze_10, relu_1, convolution_4, squeeze_13, relu_2, convolution_5, squeeze_16, add_31, convolution_6, squeeze_19, relu_3, convolution_7, squeeze_22, relu_4, convolution_8, squeeze_25, add_47, convolution_9, squeeze_28, relu_5, convolution_10, squeeze_31, relu_6, mean, relu_7, div_1, mul_78, convolution_13, squeeze_34, add_63, convolution_14, squeeze_37, relu_8, convolution_15, squeeze_40, relu_9, mean_1, relu_10, div_2, mul_100, convolution_18, squeeze_43, add_80, convolution_19, squeeze_46, relu_11, convolution_20, squeeze_49, relu_12, mean_2, relu_13, div_3, mul_122, convolution_23, squeeze_52, add_97, convolution_24, squeeze_55, clone_1, div_4, convolution_25, squeeze_58, clone_2, div_5, convolution_26, squeeze_61, add_114, convolution_27, squeeze_64, clone_3, div_6, convolution_28, squeeze_67, clone_4, div_7, convolution_29, squeeze_70, add_132, convolution_30, squeeze_73, clone_5, div_8, convolution_31, squeeze_76, clone_6, div_9, convolution_32, squeeze_79, add_150, convolution_33, squeeze_82, clone_7, div_10, convolution_34, squeeze_85, clone_8, div_11, convolution_35, squeeze_88, add_168, convolution_36, squeeze_91, clone_9, div_12, convolution_37, squeeze_94, clone_10, div_13, mean_3, relu_14, div_14, mul_238, convolution_40, squeeze_97, add_186, convolution_41, squeeze_100, clone_11, div_15, convolution_42, squeeze_103, clone_12, div_16, mean_4, relu_15, div_17, mul_262, convolution_45, squeeze_106, add_205, convolution_46, squeeze_109, clone_13, div_18, convolution_47, squeeze_112, clone_14, div_19, mean_5, relu_16, div_20, mul_286, convolution_50, squeeze_115, add_223, convolution_51, squeeze_118, clone_15, div_21, convolution_52, squeeze_121, clone_16, div_22, mean_6, relu_17, div_23, mul_310, convolution_55, squeeze_124, add_242, convolution_56, squeeze_127, clone_17, div_24, convolution_57, squeeze_130, clone_18, div_25, mean_7, relu_18, div_26, mul_334, convolution_60, squeeze_133, add_261, convolution_61, squeeze_136, clone_19, mean_8, convolution_62, view_1, permute_1, unsqueeze_186, unsqueeze_198, bitwise_and, unsqueeze_210, unsqueeze_222, unsqueeze_234, bitwise_and_1, unsqueeze_246, unsqueeze_258, unsqueeze_270, bitwise_and_2, unsqueeze_282, unsqueeze_294, unsqueeze_306, bitwise_and_3, unsqueeze_318, unsqueeze_330, unsqueeze_342, bitwise_and_4, unsqueeze_354, unsqueeze_366, unsqueeze_378, unsqueeze_390, unsqueeze_402, unsqueeze_414, unsqueeze_426, unsqueeze_438, unsqueeze_450, unsqueeze_462, unsqueeze_474, unsqueeze_486, unsqueeze_498, unsqueeze_510, unsqueeze_522, bitwise_and_5, unsqueeze_534, unsqueeze_546, unsqueeze_558, bitwise_and_6, unsqueeze_570, unsqueeze_582, unsqueeze_594, bitwise_and_7, unsqueeze_606, unsqueeze_618, unsqueeze_630, unsqueeze_642, unsqueeze_654, unsqueeze_666, unsqueeze_678, unsqueeze_690, unsqueeze_702, unsqueeze_714, unsqueeze_726, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilenetv3_large_100', benchmark_compiled_module)
