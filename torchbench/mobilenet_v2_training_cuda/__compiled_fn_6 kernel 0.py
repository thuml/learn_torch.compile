
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


# kernel path: /tmp/torchinductor_youkaichao/gg/cgg463u26szk2koomsmwdhziq76kysbb7oomrurjsyrliqijjwz4.py
# Source Nodes: [], Original ATen: [aten.sum, aten.view]

triton_poi_fused_sum_view_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_view_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (1000 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (2000 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (3000 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/fg/cfgiolyk27dnkgncan2z3my4i6rd2g6lmcjk6gdiw2og36h4w65q.py
# Source Nodes: [], Original ATen: [aten.div, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_div_hardtanh_backward_native_batch_norm_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_hardtanh_backward_native_batch_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1280
    x1 = (xindex // 1280)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1280*r2) + (125440*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (x0 + (1280*(r2 // 49)) + (2560*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (1280*r2) + (125440*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 49.0
        tmp3 = tmp1 / tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp5 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i3/ci3xyor6t7rfseb3lnqwfnwkwidcnnwsjqgcj3pbrrq267s77xqu.py
# Source Nodes: [], Original ATen: [aten.div, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_div_hardtanh_backward_native_batch_norm_backward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_hardtanh_backward_native_batch_norm_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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


# kernel path: /tmp/torchinductor_youkaichao/gd/cgdditjaxudplo4xwrl644n5liml43amdlthgp74uteuh5agrq2q.py
# Source Nodes: [], Original ATen: [aten.div, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_div_hardtanh_backward_native_batch_norm_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_hardtanh_backward_native_batch_norm_backward_3', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1280*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5w/c5wngpsmjbialvtb5dy5a3lo44tqifhmzkx2dckcm36blnyqgfng.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_div_hardtanh_backward_native_batch_norm_backward_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_hardtanh_backward_native_batch_norm_backward_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1280
    x2 = (xindex // 62720)
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (1280*x2)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp5 * tmp11
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ul/culugl4m2r35yarrzs5ozog4whco432kn75qo3rfo35duz3tpqok.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (15680*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uk/cukmioqw5vda3fymwj7afb5zp5ipwxyblbpojm4ss5d43wdvqkru.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 320
    x1 = (xindex // 320)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (15680*(r2 // 49)) + (31360*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (320*r2) + (31360*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v6/cv6nhgnez6m5333rkpzafsvyaxgafx4pt3virs3b5ac52yfvxace.py
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
    size_hints=[512, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_7', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (320*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zs/czsn6tq5wnsoojmkxfxx4fepmi5knyhi7d2zziepiar2qi66y7tf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 320
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 1e-05
    tmp3 = tmp1 + tmp2
    tmp4 = tl.math.rsqrt(tmp3)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zf/czfuk3wjvhcptxcowrajt3czxf5qe5l56gvuia7lj7avfi2sckla.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 960
    x1 = (xindex // 960)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (960*r2) + (94080*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((49*x0) + (47040*(r2 // 49)) + (94080*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (960*r2) + (94080*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/l2/cl2nmsnlj2e3bolpnypqbt4yipbqda5ktn626wtsqllqrppt4cdi.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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


# kernel path: /tmp/torchinductor_youkaichao/6w/c6wztvpyqtrfmf5jhz4yicepgh4ezxrmb5osj5ctkdgpafgmrqua.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dn/cdn3ao5c4jridaumwxqj4y64b6y64t6nr6pf4hwicvqvmsasbyrt.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (960*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (49*x2) + (47040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x2 + (960*y3)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mb/cmbnsxqarsell4mhwiutqiqj6aqbinbb5dvtxqeq3zig44vvugeh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 31360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 160
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 1e-05
    tmp3 = tmp1 + tmp2
    tmp4 = tl.math.rsqrt(tmp3)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hg/chgegsprkblha32ytcdzjm22ccnmicjsgsxz6iofjfqvyspm6til.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 31360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 160
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.math.rsqrt(tmp5)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 * tmp8
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jx/cjx3ozs3l3qtkfm3aqm2vygrjdrqyhg72egsgoakpzkppfmu3nq2.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_15', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (7840*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (49*x0) + (7840*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (x0 + (160*r3)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (r1 + (49*x0) + (7840*r2)), rmask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr5 + (x0 + (160*r3)), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = tmp6 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp27 = tmp25 - tmp26
    tmp28 = tmp20 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp18 * tmp36
    tmp39 = tmp38 + tmp34
    tmp40 = tl.math.rsqrt(tmp39)
    tmp41 = tmp32 * tmp40
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp37, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp41, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ni/cnilfgoa35o3heyu57c5v7dvcbrzmebrnzq4igqvcgbda4szmglb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 320
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
        tmp1 = tl.load(in_ptr1 + (x0 + (160*r2) + (15680*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ct/cct54whz7p3gweezxsgf3uzt4petutszm5ysc4p2rvbp23hly5j7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_17', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4x/c4xjvnirzuqr4kbior224xzlwi5s2bhwe4qajster5ibdxgnttwp.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 31360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 160
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/es/ceschzd2n73627fix4ptghyg5g6ef7zhonqedfk7pqrdx7erqcae.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 576
    x1 = (xindex // 576)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (576*r2) + (56448*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((49*x0) + (28224*(r2 // 49)) + (56448*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (576*r2) + (56448*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/gt/cgtdsegbvnsn4jqubh6er7cv6tgdtt74wynortdwnl7bt57wdptd.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (576*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eb/ceb5jfsysanqpkbdjmnfdizddm6mf2dttvhrds242vvtzllu7cdh.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_21', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (576*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/se/csercyhrave6ji4gawlmjvaoh7ooxnhjlbuboy7swoyvflczf32y.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
    xnumel = 576
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
    tmp0 = tl.load(in_ptr0 + (x2 + (576*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (49*x2) + (28224*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x2 + (576*y3)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bf/cbfcwcbljm2jrnn46c6jdtyqo25e2vumgjwr5oxo2ezul5jqlbd2.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4032
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (576*r2) + (64512*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (112896*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kg/ckg7u6efcyk3zf33q4fqkjyszpifvdaswyk734duowztlbqjnjav.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oy/coyepk4b55iuvdn22a3emjlhhfwuqwk32he6idy2mvrz5iq6ecqo.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4032
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 576
    x1 = (xindex // 576)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (576*r2) + (64512*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (112896*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (576*r2) + (64512*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/36/c36ru5oaqzkbcqp2ultlxrywchqwosgi6atlfuw2rqrlhszgoyw5.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_26', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp0 = tl.load(in_ptr0 + (x0 + (576*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33ngeegeqqb7aem2yfrde22rjx7swqdsqbuts6bcivxdypf5jlc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 576
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
    tmp0 = tl.load(in_ptr0 + (x2 + (576*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (196*x2) + (112896*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x2 + (576*y3)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oj/coj2wzr3gtbapkk4wp3asxagmvlyrjr2a5xypako2nvy3lxa3exb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 96
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 1e-05
    tmp3 = tmp1 + tmp2
    tmp4 = tl.math.rsqrt(tmp3)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oz/cozahpylh726zfx2e6toiyhvzw6vjek5apr5ifpbriqgavwejrwa.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 96
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.math.rsqrt(tmp5)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 * tmp8
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t5/ct53sr6weqfsqqrxkj5hfjslks7b7av3ilc4e44ux4fc6xwqbrhv.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_30', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 96
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 196
    r2 = (rindex // 196)
    x0 = xindex
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (18816*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (196*x0) + (18816*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (x0 + (96*r3)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (r1 + (196*x0) + (18816*r2)), rmask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr5 + (x0 + (96*r3)), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = tmp6 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp27 = tmp25 - tmp26
    tmp28 = tmp20 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp18 * tmp36
    tmp39 = tmp38 + tmp34
    tmp40 = tl.math.rsqrt(tmp39)
    tmp41 = tmp32 * tmp40
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp37, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp41, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fg/cfghdwqrwkaqcvpx36tvsbdtsvpkmfqsat2k4ga77jt7vaivs2hy.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (18816*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (96*r2) + (10752*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yj/cyjabexxvay6petm425hp4r2755dcvd5luc6rcahzmmkgwgjypol.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_32', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rz/crzf7l3syfgyenndbvpcveyrjgciiskpl5ahvavilqosb3a3jq6f.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4032
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 576
    x1 = (xindex // 576)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (576*r2) + (64512*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (112896*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (576*r2) + (64512*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/77/c77r46mn2tjha3nqnzcch5m7dazjqjsllbq5e5knougzp2vy4j23.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 96
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sw/cswr5phk3fttqak5citdeftclitpoqf6atzzpkqzky6gdc5nod4i.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2688
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (384*r2) + (43008*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (75264*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cd/ccdg7alb5pa37f52vqvizvbgitsdxrz64wtx3z5vbjt6xcvxgefb.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
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


# kernel path: /tmp/torchinductor_youkaichao/ay/caysmmkv52j2haquwrly52psogrmp7urc3zk6jrrz55rprun3gev.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2688
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (43008*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (75264*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (384*r2) + (43008*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/zk/czkkqcl2ipev7tp7futwcklafbkei4dkonxjogufsgih6hfjhniu.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_38', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3y/c3ytlfvwckzyjwo2h5he647eokvhdygzwkmn6urtn2lylablyxld.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3zkh6p5jy3w4gff4dfm434dkoxor56y26wzvcdqmcca2maincq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_40', 'mutated_arg_names': []}
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
    r1 = rindex % 196
    r2 = (rindex // 196)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (12544*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5u/c5uyvas7qrv5f57dio3pgtubbspsjlf3z54nlqdljmh4uptahpn6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (12544*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*r2) + (7168*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ko/ckoaqvornr2cqllhkotx65tcnapiwcssbn3xm6vpihybmhpibcvx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_42', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2a/c2akhf4i5owqlknnxz3of4z4x53tak73i6dfcfqv36ppih3zfnp4.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 1e-05
    tmp3 = tmp1 + tmp2
    tmp4 = tl.math.rsqrt(tmp3)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ly/clytwmxbkaux56r63zmxnrjmh5qu2fa7xbf5s2lpa2hkgzttkqqz.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.math.rsqrt(tmp5)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 * tmp8
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cf/ccfjplywhzoik3hiqhiinhykmp2deaiweraqzfwifn4dwd24vr6k.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7b/c7bklgwzle6d4bqnmrp7totwypgbb2ntpcirxqejtwszeqhavuc7.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: 'i32', 20: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(19, 20))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_46', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 196
    r2 = (rindex // 196)
    x0 = xindex
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (12544*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (12544*r2)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0 + (64*r3)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (r1 + (196*x0) + (12544*r2)), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr5 + (x0 + (64*r3)), rmask & xmask, other=0.0)
    tmp22 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (r1 + (196*x0) + (12544*r2)), rmask & xmask, other=0.0)
    tmp35 = tl.load(in_ptr8 + (x0 + (64*r3)), rmask & xmask, other=0.0)
    tmp36 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr12 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp9 = tmp7 - tmp8
    tmp10 = tmp2 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tmp2 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp23 = tmp21 - tmp22
    tmp24 = tmp16 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp30 = tmp16 + tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp37 = tmp35 - tmp36
    tmp38 = tmp30 * tmp37
    tmp39 = tl.broadcast_to(tmp38, [RBLOCK])
    tmp41 = tl.where(rmask & xmask, tmp39, 0)
    tmp42 = triton_helpers.promote_to_tensor(tl.sum(tmp41, 0))
    tmp44 = 1e-05
    tmp45 = tmp43 + tmp44
    tmp46 = tl.math.rsqrt(tmp45)
    tmp47 = tmp14 * tmp46
    tmp49 = tmp48 + tmp44
    tmp50 = tl.math.rsqrt(tmp49)
    tmp51 = tmp28 * tmp50
    tmp53 = tmp52 + tmp44
    tmp54 = tl.math.rsqrt(tmp53)
    tmp55 = tmp42 * tmp54
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp47, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp51, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp55, xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
    tl.store(out_ptr2 + (x0), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cj/ccjp2wijagkccjrmaoifig2t32epdpf4vxswfdoawaw4hblaqdea.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2688
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (43008*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (75264*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (384*r2) + (43008*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/hn/chn5crqq5jrnpgsjw2p4umywy7g5oydybvgjxkrymje43bl7rflc.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp5 = tl.load(in_ptr2 + (x3), xmask)
    tmp7 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uc/cucqzqm2gyrgpkj5ewcqrszttad77h2ctqxqwfybkckq4twn4uvs.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1344
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (192*r2) + (21504*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (37632*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tj/ctjndhxalkfme3gbkgfhsr7jnugtfge4r5kfcinyvhgal7ihbsxt.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
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


# kernel path: /tmp/torchinductor_youkaichao/45/c45vzuy7twr4hsmesgt2or75te4jsymsfdxnob7qmbpk3n7vpcth.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1344
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (21504*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (37632*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (192*r2) + (21504*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/z4/cz4bttyawa7wbhspyepayxddhxjakgmzsb2ztq4mvclu6vk7f6id.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_52', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 7
    RBLOCK: tl.constexpr = 8
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qc/cqcoosaem4c2ngfnyg5lvl2mrzilkc5stlwlcm3vo4nlkf42c5ou.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
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
    tmp1 = tl.load(in_ptr1 + (y0 + (196*x2) + (37632*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oi/coisj2dwen57feogds6dqnrkgk5yz4775qjl25zd4q3hv32gfyh3.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 25
    x1 = (xindex // 25)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x0)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (192*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((784*x1) + (150528*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ds/cdsoayx6fdqgcsafbkyasvj4oxtuhai5rohttc5rpkkualnd7b6h.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
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


# kernel path: /tmp/torchinductor_youkaichao/nv/cnv7sivawekuuhamu7yz4l2ncshh4zuz2zvuxmgzr3eboqsxgrmv.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4800
    rnumel = 126
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
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((784*x0) + (150528*(((r2 + (126*x1)) // 784) % 4)) + ((r2 + (126*x1)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.load(in_ptr2 + (x0 + (192*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/l5/cl5jvyyafjplvoxnnehiawczzu3kvhgtdq7npdp64tpkhbdu3t4l.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_57', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 25
    RBLOCK: tl.constexpr = 32
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4s/c4sjp27g5lq3is3xp7ht2xurllpwbsdjwtp63rwhmdjb52biluaj.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (784*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ah/cahkdihmqoewlnou4aidc4entactuxj2qfge35nynzwfpfsi2co7.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 32
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = 1e-05
    tmp3 = tmp1 + tmp2
    tmp4 = tl.math.rsqrt(tmp3)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oi/coicdlketihe6tyejkixj7ou4le3t24ybminzqjyu5ihggryteni.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 32
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.math.rsqrt(tmp5)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 * tmp8
    tl.store(out_ptr0 + (x3), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/um/cum73w6mabu22a4f3mjwn2syp6lluyxvqk2i5x73i33jryota5u3.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_61', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp22 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (784*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (32*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr4 + (r1 + (784*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr5 + (x0 + (32*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp5 = tmp0 + tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp5 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
        tmp17 = tmp5 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
        tmp23 = tmp21 - tmp22
        tmp24 = tmp17 * tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp19, xmask)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp28 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = tl.math.rsqrt(tmp30)
    tmp32 = tmp14 * tmp31
    tmp34 = tmp33 + tmp29
    tmp35 = tl.math.rsqrt(tmp34)
    tmp36 = tmp26 * tmp35
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp32, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cu/ccu23a27fxdhtn7xjslsmduxd45zjuvq3s3pv3gskwx3bkhgiiyc.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 25
    x1 = (xindex // 25)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x0)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((784*x1) + (25088*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (32*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/m7/cm7f5tg25otjmvslzpbnn6sbjoa3haneryl4klwczci5ambqs5eq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_63', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u6/cu6fbqnt3x54bmb7cdo3u24egtlj4akdg6h3g3242gn3dnurcvze.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4800
    rnumel = 126
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
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((784*x0) + (150528*(((r2 + (126*x1)) // 784) % 4)) + ((r2 + (126*x1)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.load(in_ptr2 + (x0 + (192*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/rr/crrw7txfelxoaqd7n6czkt6u3s6awzayq4uszinlkwool3gffbpt.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_65', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 32
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/et/cetpjtdwcpuourd25vtwo3iywewqasz5rgjzyvvgezxeex2ud2t3.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3600
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 25
    x1 = (xindex // 25)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x0)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (144*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((784*x1) + (112896*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/he/chepchwi2cl3ttpjhs62b2w6u766uyv4cjyhi7hhyu72422lklfg.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 144
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


# kernel path: /tmp/torchinductor_youkaichao/vx/cvxcvhusfmik3bp3hmxcbargkgbeqp5x4zopdznrrybcxyjow5a3.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3600
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 144)
    x0 = xindex % 144
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (144*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((784*x0) + (112896*(((r2 + (126*x1)) // 784) % 4)) + ((r2 + (126*x1)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.load(in_ptr2 + (x0 + (144*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qv/cqvj2k3pzczwpst3melrapyrpfszdchik3ahbc4ho24mdhooe45b.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_69', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (144*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g4/cg4gfkftwsh4vln3u7ewvco6ve4xxvascu2hata73a7li5ip7daa.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 144
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
    tmp0 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (784*x2) + (112896*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x2 + (144*y3)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/st/cstdhn5ytfa2o6gc47a4gbz43gwerqofaarw3frogbgnzf2gisk7.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14112
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 98
    x1 = (xindex // 98)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (144*r2) + (18432*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((3136*x1) + (451584*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2b/c2bz6jxafkjmxxq5zxakh6msyqpzkhtg6f3woqwoimrs4v4ycykn.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 98
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (98*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ca/ccap5qhkunb2hx766o3cqvglb4ubf6kwohg7aunvnp4ijsno4ofk.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14112
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 144
    x1 = (xindex // 144)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((3136*x0) + (451584*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/hj/chjp5ydsmch7xq573jkow7ecn42mqqrv37aqvkammjnhcszq77vl.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_74', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (144*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp8 = tmp2 * tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2x/c2xwrcgnnkp32xitnyqnzixgryctreitfoels4gbozyfzzdazl7m.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 144
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
    tmp0 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (3136*x2) + (451584*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x2 + (144*y3)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ug/cuguzs3dervhytu45pm5pkaknf6tybo4zm27nvlhdkmfqw5f3ee4.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 24
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = 1e-05
    tmp3 = tmp1 + tmp2
    tmp4 = tl.math.rsqrt(tmp3)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/o4/co4wpb323o3ygpidwep7rpvq2oyjqdwismt2p2inru4v5p4yj4qp.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (24*r2) + (150528*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp5 = tmp0 + tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp5 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qi/cqi2dieip2w22hy6bxzeos5q6h5deyf446c2aw5o2ilbxoyetgms.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_78 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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


# kernel path: /tmp/torchinductor_youkaichao/nn/cnnvhomzonjxzruq6lqfdbcna4vaz32njwusdqfavdpxra6jqclf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_79', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 98
    x1 = (xindex // 98)
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


# kernel path: /tmp/torchinductor_youkaichao/cz/cczve2xjys7s5vmx4l7bwr3ed4ui7xu6z4oaq4yplrry7nxtnblx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_80 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_80', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 98
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (98*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fy/cfyajemqusosr7osp3i4j66cwopm4kdkkvodd6s5baneu36vvspw.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14112
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 144
    x1 = (xindex // 144)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((3136*x0) + (451584*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/jg/cjglrf2s74uov6o4emfcpcfielbrlj7k2p3l3bzm5oijkkmfclnt.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_82 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_82', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bb/cbbai65dmfzipdad2vyak7ucgjl7wpfpm7u5idyjjlta2tdh3a5w.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_83', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 24
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.math.rsqrt(tmp5)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 * tmp8
    tl.store(in_out_ptr0 + (x3), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4f/c4fmfpl7r5jbkymzttae32bkqb257qppozp4j2nkhruxsusounsp.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_84', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 98
    x1 = (xindex // 98)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (96*r2) + (12288*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((3136*x1) + (301056*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ll/clllcvhts4llurx5u3ifag3pqtah5d3nzgogx25o64t7d2jvgzfl.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_85 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 98
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (98*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fp/cfp43xn5cnb4hobhsgffkqve3aqwnu46svpbxerbwh2436eelyjh.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_86 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_86', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 96
    x1 = (xindex // 96)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((3136*x0) + (301056*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/oj/cojxeszkmgv53af7bhmqdsqueqqvh4hds7bvcnksjirz2fg24vil.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_87 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_87', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 98
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
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp8 = tmp2 * tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/km/ckmrrhyqbcjckeubdnbu6r2m4amybbuwinzysriwi3wasv24wqyu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_88 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
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
    tmp0 = tl.load(in_ptr0 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (3136*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x2 + (96*y3)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p2/cp2fatvfopkzoftfey4i5vdm3hh2jgzz3nnoqjcn5i7hsp7sh7me.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_89 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 392
    x1 = (xindex // 392)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (96*r2) + (12288*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((12544*x1) + (1204224*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6b/c6bd7riggsb2uljswngntuuqogjayuqil6ph6nrfvltkaii7uebs.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_90 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 96
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (392*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iz/cizezg64wt7b6h6jyezckcbumxp37wripw2b5iemjjgx5s2zjchp.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_91 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 96
    x1 = (xindex // 96)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((12544*x0) + (1204224*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/nn/cnnauofdfxzgxqvgkiqywrhoo34hyign322oce2aq237267rwnyx.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_92 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_92', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 392
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
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp8 = tmp2 * tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/to/ctozew3ulesy534uqm2qdlyrpl2akogass67hz3b56og4i2ppcx4.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_93 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 50176
    xnumel = 96
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
    tmp0 = tl.load(in_ptr0 + (x2 + (96*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (12544*x2) + (1204224*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x2 + (96*y3)), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5l/c5ldwohvlxhea35sjd6urqjg4pxveqzr7btyiliz7b4m5qwdyhpk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_94 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_94', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 112
    rnumel = 7168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((112*(((r2 + (7168*x0)) // 112) % 112)) + (12544*x1) + (200704*((r2 + (7168*x0)) // 12544)) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v5/cv52nvychaxyqmy4cha4wiubhi6apm5zzrtt4zslj3exqxcgajro.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_95 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# kernel path: /tmp/torchinductor_youkaichao/4z/c4zdujmwfqqfamrbzaca6bod26kknpoplop5lt3rqkrzp3xujl4i.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_96', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 392
    x1 = (xindex // 392)
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


# kernel path: /tmp/torchinductor_youkaichao/3q/c3q73k2yojavjwaicxpcu4lqahrrluyulrvvzzhto5zgm7htzm3v.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_97 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_97', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
    xnumel = 16
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (392*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aj/cajrdj3mpdz7eymeku3mpblgo6c25wdqqvc7tpfy6nki7deynrtu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_98', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 16
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = 1e-05
    tmp3 = tmp1 + tmp2
    tmp4 = tl.math.rsqrt(tmp3)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ch/cchuwddkh27etvvycfv6ofnxiyxrp4r6okuipwysqttmcik64lgy.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_99 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_99', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 392
    x1 = (xindex // 392)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (32*r2) + (4096*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((12544*x1) + (401408*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bl/cblo6ofy35whosjhtytvhoqtdnu7lpseushdb72twvsmyc4dwu7l.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_100 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_100', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 32
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (392*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3q/c3qovdmistjjjxnmehvsbd64k3mwqv5lhdqrrnrdaiwhqxwffzfj.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_101 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_101', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((12544*x0) + (401408*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/to/cto42cxmnm2xnih2aw37snb66vvssmgguc4s5kldinmycpr247xe.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_102 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_102', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 392
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
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp8 = tmp2 * tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/li/cliv2lpfz5mzn6i233ojaymhkc2b3cotsazmrdb4xvl7znzczpdj.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_103 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_103', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 50176
    xnumel = 32
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
    tmp0 = tl.load(in_ptr0 + (x2 + (32*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (12544*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x2 + (32*y3)), tmp10, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_159, primals_160, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, convolution, clamp_max, convolution_1, clamp_max_1, convolution_2, add_5, convolution_3, clamp_max_2, convolution_4, clamp_max_3, convolution_5, add_11, convolution_6, clamp_max_4, convolution_7, clamp_max_5, convolution_8, add_18, convolution_9, clamp_max_6, convolution_10, clamp_max_7, convolution_11, add_24, convolution_12, clamp_max_8, convolution_13, clamp_max_9, convolution_14, add_31, convolution_15, clamp_max_10, convolution_16, clamp_max_11, convolution_17, add_38, convolution_18, clamp_max_12, convolution_19, clamp_max_13, convolution_20, add_44, convolution_21, clamp_max_14, convolution_22, clamp_max_15, convolution_23, add_51, convolution_24, clamp_max_16, convolution_25, clamp_max_17, convolution_26, add_58, convolution_27, clamp_max_18, convolution_28, clamp_max_19, convolution_29, add_65, convolution_30, clamp_max_20, convolution_31, clamp_max_21, convolution_32, add_71, convolution_33, clamp_max_22, convolution_34, clamp_max_23, convolution_35, add_78, convolution_36, clamp_max_24, convolution_37, clamp_max_25, convolution_38, add_85, convolution_39, clamp_max_26, convolution_40, clamp_max_27, convolution_41, add_91, convolution_42, clamp_max_28, convolution_43, clamp_max_29, convolution_44, add_98, convolution_45, clamp_max_30, convolution_46, clamp_max_31, convolution_47, add_105, convolution_48, clamp_max_32, convolution_49, clamp_max_33, convolution_50, add_111, convolution_51, clone_35, permute_1, bitwise_or, bitwise_or_1, bitwise_or_2, bitwise_or_3, bitwise_or_4, bitwise_or_5, bitwise_or_6, bitwise_or_7, bitwise_or_8, bitwise_or_9, bitwise_or_10, bitwise_or_11, bitwise_or_12, bitwise_or_13, bitwise_or_14, bitwise_or_15, bitwise_or_16, bitwise_or_17, bitwise_or_18, bitwise_or_19, bitwise_or_20, bitwise_or_21, bitwise_or_22, bitwise_or_23, bitwise_or_24, bitwise_or_25, bitwise_or_26, bitwise_or_27, bitwise_or_28, bitwise_or_29, bitwise_or_30, bitwise_or_31, bitwise_or_32, bitwise_or_33, bitwise_or_34, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_4, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_7, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_8, (16, ), (1, ))
    assert_size_stride(primals_10, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_11, (96, ), (1, ))
    assert_size_stride(primals_13, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_14, (96, ), (1, ))
    assert_size_stride(primals_16, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_19, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_20, (144, ), (1, ))
    assert_size_stride(primals_22, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (144, ), (1, ))
    assert_size_stride(primals_25, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_26, (24, ), (1, ))
    assert_size_stride(primals_28, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_29, (144, ), (1, ))
    assert_size_stride(primals_31, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_32, (144, ), (1, ))
    assert_size_stride(primals_34, (32, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_35, (32, ), (1, ))
    assert_size_stride(primals_37, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_38, (192, ), (1, ))
    assert_size_stride(primals_40, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_41, (192, ), (1, ))
    assert_size_stride(primals_43, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_44, (32, ), (1, ))
    assert_size_stride(primals_46, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_47, (192, ), (1, ))
    assert_size_stride(primals_49, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_50, (192, ), (1, ))
    assert_size_stride(primals_52, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_53, (32, ), (1, ))
    assert_size_stride(primals_55, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_56, (192, ), (1, ))
    assert_size_stride(primals_58, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_59, (192, ), (1, ))
    assert_size_stride(primals_61, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_62, (64, ), (1, ))
    assert_size_stride(primals_64, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_65, (384, ), (1, ))
    assert_size_stride(primals_67, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_68, (384, ), (1, ))
    assert_size_stride(primals_70, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_71, (64, ), (1, ))
    assert_size_stride(primals_73, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_74, (384, ), (1, ))
    assert_size_stride(primals_76, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_77, (384, ), (1, ))
    assert_size_stride(primals_79, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_80, (64, ), (1, ))
    assert_size_stride(primals_82, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_83, (384, ), (1, ))
    assert_size_stride(primals_85, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_86, (384, ), (1, ))
    assert_size_stride(primals_88, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_89, (64, ), (1, ))
    assert_size_stride(primals_91, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_92, (384, ), (1, ))
    assert_size_stride(primals_94, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_95, (384, ), (1, ))
    assert_size_stride(primals_97, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_98, (96, ), (1, ))
    assert_size_stride(primals_100, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_101, (576, ), (1, ))
    assert_size_stride(primals_103, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_104, (576, ), (1, ))
    assert_size_stride(primals_106, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_107, (96, ), (1, ))
    assert_size_stride(primals_109, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_110, (576, ), (1, ))
    assert_size_stride(primals_112, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_113, (576, ), (1, ))
    assert_size_stride(primals_115, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_116, (96, ), (1, ))
    assert_size_stride(primals_118, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_119, (576, ), (1, ))
    assert_size_stride(primals_121, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_122, (576, ), (1, ))
    assert_size_stride(primals_124, (160, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_125, (160, ), (1, ))
    assert_size_stride(primals_127, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_128, (960, ), (1, ))
    assert_size_stride(primals_130, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_131, (960, ), (1, ))
    assert_size_stride(primals_133, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_134, (160, ), (1, ))
    assert_size_stride(primals_136, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_137, (960, ), (1, ))
    assert_size_stride(primals_139, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_140, (960, ), (1, ))
    assert_size_stride(primals_142, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_143, (160, ), (1, ))
    assert_size_stride(primals_145, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_146, (960, ), (1, ))
    assert_size_stride(primals_148, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_149, (960, ), (1, ))
    assert_size_stride(primals_151, (320, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_152, (320, ), (1, ))
    assert_size_stride(primals_154, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_155, (1280, ), (1, ))
    assert_size_stride(primals_159, (32, ), (1, ))
    assert_size_stride(primals_160, (32, ), (1, ))
    assert_size_stride(primals_162, (32, ), (1, ))
    assert_size_stride(primals_163, (32, ), (1, ))
    assert_size_stride(primals_165, (16, ), (1, ))
    assert_size_stride(primals_166, (16, ), (1, ))
    assert_size_stride(primals_168, (96, ), (1, ))
    assert_size_stride(primals_169, (96, ), (1, ))
    assert_size_stride(primals_171, (96, ), (1, ))
    assert_size_stride(primals_172, (96, ), (1, ))
    assert_size_stride(primals_174, (24, ), (1, ))
    assert_size_stride(primals_175, (24, ), (1, ))
    assert_size_stride(primals_177, (144, ), (1, ))
    assert_size_stride(primals_178, (144, ), (1, ))
    assert_size_stride(primals_180, (144, ), (1, ))
    assert_size_stride(primals_181, (144, ), (1, ))
    assert_size_stride(primals_183, (24, ), (1, ))
    assert_size_stride(primals_184, (24, ), (1, ))
    assert_size_stride(primals_186, (144, ), (1, ))
    assert_size_stride(primals_187, (144, ), (1, ))
    assert_size_stride(primals_189, (144, ), (1, ))
    assert_size_stride(primals_190, (144, ), (1, ))
    assert_size_stride(primals_192, (32, ), (1, ))
    assert_size_stride(primals_193, (32, ), (1, ))
    assert_size_stride(primals_195, (192, ), (1, ))
    assert_size_stride(primals_196, (192, ), (1, ))
    assert_size_stride(primals_198, (192, ), (1, ))
    assert_size_stride(primals_199, (192, ), (1, ))
    assert_size_stride(primals_201, (32, ), (1, ))
    assert_size_stride(primals_202, (32, ), (1, ))
    assert_size_stride(primals_204, (192, ), (1, ))
    assert_size_stride(primals_205, (192, ), (1, ))
    assert_size_stride(primals_207, (192, ), (1, ))
    assert_size_stride(primals_208, (192, ), (1, ))
    assert_size_stride(primals_210, (32, ), (1, ))
    assert_size_stride(primals_211, (32, ), (1, ))
    assert_size_stride(primals_213, (192, ), (1, ))
    assert_size_stride(primals_214, (192, ), (1, ))
    assert_size_stride(primals_216, (192, ), (1, ))
    assert_size_stride(primals_217, (192, ), (1, ))
    assert_size_stride(primals_219, (64, ), (1, ))
    assert_size_stride(primals_220, (64, ), (1, ))
    assert_size_stride(primals_222, (384, ), (1, ))
    assert_size_stride(primals_223, (384, ), (1, ))
    assert_size_stride(primals_225, (384, ), (1, ))
    assert_size_stride(primals_226, (384, ), (1, ))
    assert_size_stride(primals_228, (64, ), (1, ))
    assert_size_stride(primals_229, (64, ), (1, ))
    assert_size_stride(primals_231, (384, ), (1, ))
    assert_size_stride(primals_232, (384, ), (1, ))
    assert_size_stride(primals_234, (384, ), (1, ))
    assert_size_stride(primals_235, (384, ), (1, ))
    assert_size_stride(primals_237, (64, ), (1, ))
    assert_size_stride(primals_238, (64, ), (1, ))
    assert_size_stride(primals_240, (384, ), (1, ))
    assert_size_stride(primals_241, (384, ), (1, ))
    assert_size_stride(primals_243, (384, ), (1, ))
    assert_size_stride(primals_244, (384, ), (1, ))
    assert_size_stride(primals_246, (64, ), (1, ))
    assert_size_stride(primals_247, (64, ), (1, ))
    assert_size_stride(primals_249, (384, ), (1, ))
    assert_size_stride(primals_250, (384, ), (1, ))
    assert_size_stride(primals_252, (384, ), (1, ))
    assert_size_stride(primals_253, (384, ), (1, ))
    assert_size_stride(primals_255, (96, ), (1, ))
    assert_size_stride(primals_256, (96, ), (1, ))
    assert_size_stride(primals_258, (576, ), (1, ))
    assert_size_stride(primals_259, (576, ), (1, ))
    assert_size_stride(primals_261, (576, ), (1, ))
    assert_size_stride(primals_262, (576, ), (1, ))
    assert_size_stride(primals_264, (96, ), (1, ))
    assert_size_stride(primals_265, (96, ), (1, ))
    assert_size_stride(primals_267, (576, ), (1, ))
    assert_size_stride(primals_268, (576, ), (1, ))
    assert_size_stride(primals_270, (576, ), (1, ))
    assert_size_stride(primals_271, (576, ), (1, ))
    assert_size_stride(primals_273, (96, ), (1, ))
    assert_size_stride(primals_274, (96, ), (1, ))
    assert_size_stride(primals_276, (576, ), (1, ))
    assert_size_stride(primals_277, (576, ), (1, ))
    assert_size_stride(primals_279, (576, ), (1, ))
    assert_size_stride(primals_280, (576, ), (1, ))
    assert_size_stride(primals_282, (160, ), (1, ))
    assert_size_stride(primals_283, (160, ), (1, ))
    assert_size_stride(primals_285, (960, ), (1, ))
    assert_size_stride(primals_286, (960, ), (1, ))
    assert_size_stride(primals_288, (960, ), (1, ))
    assert_size_stride(primals_289, (960, ), (1, ))
    assert_size_stride(primals_291, (160, ), (1, ))
    assert_size_stride(primals_292, (160, ), (1, ))
    assert_size_stride(primals_294, (960, ), (1, ))
    assert_size_stride(primals_295, (960, ), (1, ))
    assert_size_stride(primals_297, (960, ), (1, ))
    assert_size_stride(primals_298, (960, ), (1, ))
    assert_size_stride(primals_300, (160, ), (1, ))
    assert_size_stride(primals_301, (160, ), (1, ))
    assert_size_stride(primals_303, (960, ), (1, ))
    assert_size_stride(primals_304, (960, ), (1, ))
    assert_size_stride(primals_306, (960, ), (1, ))
    assert_size_stride(primals_307, (960, ), (1, ))
    assert_size_stride(primals_309, (320, ), (1, ))
    assert_size_stride(primals_310, (320, ), (1, ))
    assert_size_stride(primals_312, (1280, ), (1, ))
    assert_size_stride(primals_313, (1280, ), (1, ))
    assert_size_stride(primals_315, (4, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(clamp_max, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_1, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(clamp_max_1, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_2, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(add_5, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_3, (4, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(clamp_max_2, (4, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(convolution_4, (4, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(clamp_max_3, (4, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(convolution_5, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(add_11, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_6, (4, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(clamp_max_4, (4, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(convolution_7, (4, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(clamp_max_5, (4, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(convolution_8, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(add_18, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_9, (4, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(clamp_max_6, (4, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(convolution_10, (4, 144, 28, 28), (112896, 1, 4032, 144))
    assert_size_stride(clamp_max_7, (4, 144, 28, 28), (112896, 1, 4032, 144))
    assert_size_stride(convolution_11, (4, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(add_24, (4, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(convolution_12, (4, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(clamp_max_8, (4, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_13, (4, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(clamp_max_9, (4, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_14, (4, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(add_31, (4, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(convolution_15, (4, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(clamp_max_10, (4, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_16, (4, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(clamp_max_11, (4, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_17, (4, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(add_38, (4, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(convolution_18, (4, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(clamp_max_12, (4, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_19, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(clamp_max_13, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_20, (4, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(add_44, (4, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(convolution_21, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(clamp_max_14, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_22, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(clamp_max_15, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_23, (4, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(add_51, (4, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(convolution_24, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(clamp_max_16, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_25, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(clamp_max_17, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_26, (4, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(add_58, (4, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(convolution_27, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(clamp_max_18, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_28, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(clamp_max_19, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_29, (4, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(add_65, (4, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(convolution_30, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(clamp_max_20, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_31, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(clamp_max_21, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_32, (4, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(add_71, (4, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(convolution_33, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(clamp_max_22, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_34, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(clamp_max_23, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_35, (4, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(add_78, (4, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(convolution_36, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(clamp_max_24, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_37, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(clamp_max_25, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_38, (4, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(add_85, (4, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(convolution_39, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(clamp_max_26, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_40, (4, 576, 7, 7), (28224, 1, 4032, 576))
    assert_size_stride(clamp_max_27, (4, 576, 7, 7), (28224, 1, 4032, 576))
    assert_size_stride(convolution_41, (4, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(add_91, (4, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_42, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(clamp_max_28, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_43, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(clamp_max_29, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_44, (4, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(add_98, (4, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_45, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(clamp_max_30, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_46, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(clamp_max_31, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_47, (4, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(add_105, (4, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_48, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(clamp_max_32, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_49, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(clamp_max_33, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_50, (4, 320, 7, 7), (15680, 1, 2240, 320))
    assert_size_stride(add_111, (4, 320, 7, 7), (15680, 1, 2240, 320))
    assert_size_stride(convolution_51, (4, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(clone_35, (4, 1280), (1280, 1))
    assert_size_stride(permute_1, (1000, 1280), (1280, 1))
    assert_size_stride(bitwise_or, (4, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(bitwise_or_1, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(bitwise_or_2, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(bitwise_or_3, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(bitwise_or_4, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(bitwise_or_5, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(bitwise_or_6, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(bitwise_or_7, (4, 576, 7, 7), (28224, 1, 4032, 576))
    assert_size_stride(bitwise_or_8, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(bitwise_or_9, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(bitwise_or_10, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(bitwise_or_11, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(bitwise_or_12, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(bitwise_or_13, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(bitwise_or_14, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(bitwise_or_15, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(bitwise_or_16, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(bitwise_or_17, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(bitwise_or_18, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(bitwise_or_19, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(bitwise_or_20, (4, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(bitwise_or_21, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(bitwise_or_22, (4, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(bitwise_or_23, (4, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(bitwise_or_24, (4, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(bitwise_or_25, (4, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(bitwise_or_26, (4, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(bitwise_or_27, (4, 144, 28, 28), (112896, 1, 4032, 144))
    assert_size_stride(bitwise_or_28, (4, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(bitwise_or_29, (4, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(bitwise_or_30, (4, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(bitwise_or_31, (4, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(bitwise_or_32, (4, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(bitwise_or_33, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(bitwise_or_34, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), clone_35, out=buf1)
        del clone_35
        buf2 = empty((1000, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_sum_view_0.run(tangents_1, buf2, 1000, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty_strided((1280, 2), (1, 1280), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1280, 2), (1, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_div_hardtanh_backward_native_batch_norm_backward_1.run(bitwise_or, buf0, convolution_51, primals_312, buf3, buf5, 2560, 98, grid=grid(2560), stream=stream0)
        del convolution_51
        del primals_312
        buf4 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardtanh_backward_native_batch_norm_backward_2.run(buf3, buf4, 1280, 2, grid=grid(1280), stream=stream0)
        del buf3
        buf6 = empty((1280, ), device='cuda', dtype=torch.float32)
        buf7 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.div, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardtanh_backward_native_batch_norm_backward_3.run(buf7, buf5, primals_313, 1280, 2, grid=grid(1280), stream=stream0)
        del buf5
        buf8 = empty_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_div_hardtanh_backward_native_batch_norm_backward_4.run(bitwise_or, buf0, primals_313, primals_155, buf8, 250880, grid=grid(250880), stream=stream0)
        del bitwise_or
        del buf0
        del primals_155
        del primals_313
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf9 = aten.convolution_backward(buf8, add_111, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_111
        del buf8
        del primals_154
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        buf12 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_5.run(buf10, buf12, 320, 196, grid=grid(320), stream=stream0)
        buf13 = empty_strided((320, 2), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_6.run(buf10, convolution_50, primals_309, buf13, 640, 98, grid=grid(640), stream=stream0)
        del convolution_50
        del primals_309
        buf14 = empty((320, ), device='cuda', dtype=torch.float32)
        buf15 = buf14; del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_7.run(buf15, buf13, primals_310, 320, 2, grid=grid(320), stream=stream0)
        del buf13
        buf16 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_8.run(buf16, primals_310, primals_152, 62720, grid=grid(62720), stream=stream0)
        del primals_152
        del primals_310
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf17 = aten.convolution_backward(buf16, clamp_max_33, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf16
        del clamp_max_33
        del primals_151
        buf18 = buf17[0]
        buf19 = buf17[1]
        del buf17
        buf20 = empty_strided((960, 2), (1, 960), device='cuda', dtype=torch.float32)
        buf22 = empty_strided((960, 2), (1, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_9.run(bitwise_or_1, buf18, convolution_49, primals_306, buf20, buf22, 1920, 98, grid=grid(1920), stream=stream0)
        del convolution_49
        del primals_306
        buf21 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_10.run(buf20, buf21, 960, 2, grid=grid(960), stream=stream0)
        buf23 = empty((960, ), device='cuda', dtype=torch.float32)
        buf24 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_11.run(buf24, buf22, primals_307, 960, 2, grid=grid(960), stream=stream0)
        buf25 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_12.run(bitwise_or_1, buf18, primals_307, primals_149, buf25, 196, 960, grid=grid(196, 960), stream=stream0)
        del bitwise_or_1
        del buf18
        del primals_149
        del primals_307
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf26 = aten.convolution_backward(buf25, clamp_max_32, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 960, [True, True, False])
        del clamp_max_32
        del primals_148
        buf27 = buf26[0]
        buf28 = buf26[1]
        del buf26
        buf29 = buf22; del buf22  # reuse
        buf31 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_9.run(bitwise_or_2, buf27, convolution_48, primals_303, buf29, buf31, 1920, 98, grid=grid(1920), stream=stream0)
        del convolution_48
        del primals_303
        buf30 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_10.run(buf29, buf30, 960, 2, grid=grid(960), stream=stream0)
        buf32 = empty((960, ), device='cuda', dtype=torch.float32)
        buf33 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_11.run(buf33, buf31, primals_304, 960, 2, grid=grid(960), stream=stream0)
        buf34 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_12.run(bitwise_or_2, buf27, primals_304, primals_146, buf34, 196, 960, grid=grid(196, 960), stream=stream0)
        del bitwise_or_2
        del buf27
        del primals_146
        del primals_304
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf35 = aten.convolution_backward(buf34, add_105, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_105
        del primals_145
        buf36 = buf35[0]
        buf37 = buf35[1]
        del buf35
        buf42 = empty((4, 160, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_13.run(buf36, primals_301, primals_143, buf42, 31360, grid=grid(31360), stream=stream0)
        del primals_143
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf43 = aten.convolution_backward(buf42, clamp_max_31, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clamp_max_31
        del primals_142
        buf44 = buf43[0]
        buf51 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_12.run(bitwise_or_3, buf44, primals_298, primals_140, buf51, 196, 960, grid=grid(196, 960), stream=stream0)
        del primals_140
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf52 = aten.convolution_backward(buf51, clamp_max_30, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 960, [True, True, False])
        del clamp_max_30
        del primals_139
        buf53 = buf52[0]
        buf60 = buf51; del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_12.run(bitwise_or_4, buf53, primals_295, primals_137, buf60, 196, 960, grid=grid(196, 960), stream=stream0)
        del primals_137
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf61 = aten.convolution_backward(buf60, add_98, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_98
        del primals_136
        buf62 = buf61[0]
        buf67 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_14.run(buf36, buf62, primals_292, primals_134, buf67, 31360, grid=grid(31360), stream=stream0)
        del primals_134
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf68 = aten.convolution_backward(buf67, clamp_max_29, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf67
        del clamp_max_29
        del primals_133
        buf69 = buf68[0]
        buf76 = buf60; del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_12.run(bitwise_or_5, buf69, primals_289, primals_131, buf76, 196, 960, grid=grid(196, 960), stream=stream0)
        del primals_131
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf77 = aten.convolution_backward(buf76, clamp_max_28, primals_130, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 960, [True, True, False])
        del clamp_max_28
        del primals_130
        buf78 = buf77[0]
        buf85 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_12.run(bitwise_or_6, buf78, primals_286, primals_128, buf85, 196, 960, grid=grid(196, 960), stream=stream0)
        del primals_128
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf86 = aten.convolution_backward(buf85, add_91, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_91
        del buf85
        del primals_127
        buf87 = buf86[0]
        buf38 = empty((160, ), device='cuda', dtype=torch.float32)
        buf64 = empty((160, ), device='cuda', dtype=torch.float32)
        buf65 = empty((160, ), device='cuda', dtype=torch.float32)
        buf89 = empty((160, ), device='cuda', dtype=torch.float32)
        buf90 = empty((160, ), device='cuda', dtype=torch.float32)
        buf66 = buf65; del buf65  # reuse
        buf91 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_15.run(buf66, buf91, buf36, buf62, convolution_44, primals_291, buf87, convolution_41, primals_282, primals_292, primals_283, buf38, buf64, buf89, 160, 196, grid=grid(160), stream=stream0)
        del convolution_41
        del convolution_44
        del primals_282
        del primals_291
        del primals_292
        buf39 = empty_strided((160, 2), (1, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_16.run(buf36, convolution_47, primals_300, buf39, 320, 98, grid=grid(320), stream=stream0)
        del convolution_47
        del primals_300
        buf40 = empty((160, ), device='cuda', dtype=torch.float32)
        buf41 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_17.run(buf41, buf39, primals_301, 160, 2, grid=grid(160), stream=stream0)
        del buf39
        del primals_301
        buf45 = buf43[1]
        del buf43
        buf46 = buf31; del buf31  # reuse
        buf48 = buf29; del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_9.run(bitwise_or_3, buf44, convolution_46, primals_297, buf46, buf48, 1920, 98, grid=grid(1920), stream=stream0)
        del bitwise_or_3
        del buf44
        del convolution_46
        del primals_297
        buf47 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_10.run(buf46, buf47, 960, 2, grid=grid(960), stream=stream0)
        buf49 = empty((960, ), device='cuda', dtype=torch.float32)
        buf50 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_11.run(buf50, buf48, primals_298, 960, 2, grid=grid(960), stream=stream0)
        del primals_298
        buf54 = buf52[1]
        del buf52
        buf55 = buf48; del buf48  # reuse
        buf57 = buf46; del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_9.run(bitwise_or_4, buf53, convolution_45, primals_294, buf55, buf57, 1920, 98, grid=grid(1920), stream=stream0)
        del bitwise_or_4
        del buf53
        del convolution_45
        del primals_294
        buf56 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_10.run(buf55, buf56, 960, 2, grid=grid(960), stream=stream0)
        buf58 = empty((960, ), device='cuda', dtype=torch.float32)
        buf59 = buf58; del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_11.run(buf59, buf57, primals_295, 960, 2, grid=grid(960), stream=stream0)
        del primals_295
        buf63 = buf61[1]
        del buf61
        buf70 = buf68[1]
        del buf68
        buf71 = buf57; del buf57  # reuse
        buf73 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_9.run(bitwise_or_5, buf69, convolution_43, primals_288, buf71, buf73, 1920, 98, grid=grid(1920), stream=stream0)
        del bitwise_or_5
        del buf69
        del convolution_43
        del primals_288
        buf72 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_10.run(buf71, buf72, 960, 2, grid=grid(960), stream=stream0)
        buf74 = empty((960, ), device='cuda', dtype=torch.float32)
        buf75 = buf74; del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_11.run(buf75, buf73, primals_289, 960, 2, grid=grid(960), stream=stream0)
        del primals_289
        buf79 = buf77[1]
        del buf77
        buf80 = buf73; del buf73  # reuse
        buf82 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_9.run(bitwise_or_6, buf78, convolution_42, primals_285, buf80, buf82, 1920, 98, grid=grid(1920), stream=stream0)
        del bitwise_or_6
        del buf78
        del convolution_42
        del primals_285
        buf81 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_10.run(buf80, buf81, 960, 2, grid=grid(960), stream=stream0)
        del buf80
        buf83 = empty((960, ), device='cuda', dtype=torch.float32)
        buf84 = buf83; del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_11.run(buf84, buf82, primals_286, 960, 2, grid=grid(960), stream=stream0)
        del buf82
        del primals_286
        buf88 = buf86[1]
        del buf86
        buf92 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_18.run(buf92, buf62, buf87, primals_283, primals_125, 31360, grid=grid(31360), stream=stream0)
        del buf62
        del buf87
        del primals_125
        del primals_283
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf93 = aten.convolution_backward(buf92, clamp_max_27, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf92
        del clamp_max_27
        del primals_124
        buf94 = buf93[0]
        buf95 = buf93[1]
        del buf93
        buf96 = empty_strided((576, 2), (1, 576), device='cuda', dtype=torch.float32)
        buf98 = empty_strided((576, 2), (1, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_19.run(bitwise_or_7, buf94, convolution_40, primals_279, buf96, buf98, 1152, 98, grid=grid(1152), stream=stream0)
        del convolution_40
        del primals_279
        buf97 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_20.run(buf96, buf97, 576, 2, grid=grid(576), stream=stream0)
        del buf96
        buf99 = empty((576, ), device='cuda', dtype=torch.float32)
        buf100 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_21.run(buf100, buf98, primals_280, 576, 2, grid=grid(576), stream=stream0)
        del buf98
        buf101 = empty_strided((4, 576, 7, 7), (28224, 1, 4032, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_22.run(bitwise_or_7, buf94, primals_280, primals_122, buf101, 196, 576, grid=grid(196, 576), stream=stream0)
        del bitwise_or_7
        del buf94
        del primals_122
        del primals_280
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf102 = aten.convolution_backward(buf101, clamp_max_26, primals_121, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False])
        del buf101
        del clamp_max_26
        del primals_121
        buf103 = buf102[0]
        buf104 = buf102[1]
        del buf102
        buf105 = empty((576, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_23.run(bitwise_or_8, buf103, buf105, 4032, 112, grid=grid(4032), stream=stream0)
        buf106 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_24.run(buf105, buf106, 576, 7, grid=grid(576), stream=stream0)
        buf107 = reinterpret_tensor(buf105, (576, 7), (1, 576), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_25.run(bitwise_or_8, buf103, convolution_39, primals_276, buf107, 4032, 112, grid=grid(4032), stream=stream0)
        del convolution_39
        del primals_276
        buf108 = empty((576, ), device='cuda', dtype=torch.float32)
        buf109 = buf108; del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_26.run(buf109, buf107, primals_277, 576, 7, grid=grid(576), stream=stream0)
        buf110 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_27.run(bitwise_or_8, buf103, primals_277, primals_119, buf110, 784, 576, grid=grid(784, 576), stream=stream0)
        del bitwise_or_8
        del buf103
        del primals_119
        del primals_277
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf111 = aten.convolution_backward(buf110, add_85, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_85
        del primals_118
        buf112 = buf111[0]
        buf113 = buf111[1]
        del buf111
        buf118 = empty((4, 96, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_28.run(buf112, primals_274, primals_116, buf118, 75264, grid=grid(75264), stream=stream0)
        del primals_116
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf119 = aten.convolution_backward(buf118, clamp_max_25, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clamp_max_25
        del primals_115
        buf120 = buf119[0]
        buf127 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_27.run(bitwise_or_9, buf120, primals_271, primals_113, buf127, 784, 576, grid=grid(784, 576), stream=stream0)
        del primals_113
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf128 = aten.convolution_backward(buf127, clamp_max_24, primals_112, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False])
        del clamp_max_24
        del primals_112
        buf129 = buf128[0]
        buf136 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_27.run(bitwise_or_10, buf129, primals_268, primals_110, buf136, 784, 576, grid=grid(784, 576), stream=stream0)
        del primals_110
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf137 = aten.convolution_backward(buf136, add_78, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_78
        del primals_109
        buf138 = buf137[0]
        buf143 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_29.run(buf112, buf138, primals_265, primals_107, buf143, 75264, grid=grid(75264), stream=stream0)
        del primals_107
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf144 = aten.convolution_backward(buf143, clamp_max_23, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf143
        del clamp_max_23
        del primals_106
        buf145 = buf144[0]
        buf152 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_27.run(bitwise_or_11, buf145, primals_262, primals_104, buf152, 784, 576, grid=grid(784, 576), stream=stream0)
        del primals_104
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf153 = aten.convolution_backward(buf152, clamp_max_22, primals_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False])
        del clamp_max_22
        del primals_103
        buf154 = buf153[0]
        buf161 = buf152; del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_27.run(bitwise_or_12, buf154, primals_259, primals_101, buf161, 784, 576, grid=grid(784, 576), stream=stream0)
        del primals_101
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf162 = aten.convolution_backward(buf161, add_71, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_71
        del buf161
        del primals_100
        buf163 = buf162[0]
        buf114 = empty((96, ), device='cuda', dtype=torch.float32)
        buf140 = empty((96, ), device='cuda', dtype=torch.float32)
        buf141 = empty((96, ), device='cuda', dtype=torch.float32)
        buf165 = empty((96, ), device='cuda', dtype=torch.float32)
        buf166 = empty((96, ), device='cuda', dtype=torch.float32)
        buf142 = buf141; del buf141  # reuse
        buf167 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_30.run(buf142, buf167, buf112, buf138, convolution_35, primals_264, buf163, convolution_32, primals_255, primals_265, primals_256, buf114, buf140, buf165, 96, 784, grid=grid(96), stream=stream0)
        del convolution_32
        del convolution_35
        del primals_255
        del primals_264
        del primals_265
        buf115 = empty((96, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_31.run(buf112, convolution_38, primals_273, buf115, 672, 112, grid=grid(672), stream=stream0)
        del convolution_38
        del primals_273
        buf116 = empty((96, ), device='cuda', dtype=torch.float32)
        buf117 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_32.run(buf117, buf115, primals_274, 96, 7, grid=grid(96), stream=stream0)
        del buf115
        del primals_274
        buf121 = buf119[1]
        del buf119
        buf122 = reinterpret_tensor(buf107, (576, 7), (7, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_23.run(bitwise_or_9, buf120, buf122, 4032, 112, grid=grid(4032), stream=stream0)
        buf123 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_24.run(buf122, buf123, 576, 7, grid=grid(576), stream=stream0)
        buf124 = reinterpret_tensor(buf122, (576, 7), (1, 576), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_33.run(bitwise_or_9, buf120, convolution_37, primals_270, buf124, 4032, 112, grid=grid(4032), stream=stream0)
        del bitwise_or_9
        del buf120
        del convolution_37
        del primals_270
        buf125 = empty((576, ), device='cuda', dtype=torch.float32)
        buf126 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_26.run(buf126, buf124, primals_271, 576, 7, grid=grid(576), stream=stream0)
        del primals_271
        buf130 = buf128[1]
        del buf128
        buf131 = reinterpret_tensor(buf124, (576, 7), (7, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_23.run(bitwise_or_10, buf129, buf131, 4032, 112, grid=grid(4032), stream=stream0)
        buf132 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_24.run(buf131, buf132, 576, 7, grid=grid(576), stream=stream0)
        buf133 = reinterpret_tensor(buf131, (576, 7), (1, 576), 0); del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_33.run(bitwise_or_10, buf129, convolution_36, primals_267, buf133, 4032, 112, grid=grid(4032), stream=stream0)
        del bitwise_or_10
        del buf129
        del convolution_36
        del primals_267
        buf134 = empty((576, ), device='cuda', dtype=torch.float32)
        buf135 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_26.run(buf135, buf133, primals_268, 576, 7, grid=grid(576), stream=stream0)
        del primals_268
        buf139 = buf137[1]
        del buf137
        buf146 = buf144[1]
        del buf144
        buf147 = reinterpret_tensor(buf133, (576, 7), (7, 1), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_23.run(bitwise_or_11, buf145, buf147, 4032, 112, grid=grid(4032), stream=stream0)
        buf148 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_24.run(buf147, buf148, 576, 7, grid=grid(576), stream=stream0)
        buf149 = reinterpret_tensor(buf147, (576, 7), (1, 576), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_33.run(bitwise_or_11, buf145, convolution_34, primals_261, buf149, 4032, 112, grid=grid(4032), stream=stream0)
        del bitwise_or_11
        del buf145
        del convolution_34
        del primals_261
        buf150 = empty((576, ), device='cuda', dtype=torch.float32)
        buf151 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_26.run(buf151, buf149, primals_262, 576, 7, grid=grid(576), stream=stream0)
        del primals_262
        buf155 = buf153[1]
        del buf153
        buf156 = reinterpret_tensor(buf149, (576, 7), (7, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_23.run(bitwise_or_12, buf154, buf156, 4032, 112, grid=grid(4032), stream=stream0)
        buf157 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_24.run(buf156, buf157, 576, 7, grid=grid(576), stream=stream0)
        buf158 = reinterpret_tensor(buf156, (576, 7), (1, 576), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_33.run(bitwise_or_12, buf154, convolution_33, primals_258, buf158, 4032, 112, grid=grid(4032), stream=stream0)
        del bitwise_or_12
        del convolution_33
        del primals_258
        buf159 = empty((576, ), device='cuda', dtype=torch.float32)
        buf160 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_26.run(buf160, buf158, primals_259, 576, 7, grid=grid(576), stream=stream0)
        del buf158
        del primals_259
        buf164 = buf162[1]
        del buf162
        buf168 = buf112; del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_34.run(buf168, buf138, buf163, primals_256, primals_98, 75264, grid=grid(75264), stream=stream0)
        del buf138
        del buf163
        del primals_256
        del primals_98
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf169 = aten.convolution_backward(buf168, clamp_max_21, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf168
        del clamp_max_21
        del primals_97
        buf170 = buf169[0]
        buf171 = buf169[1]
        del buf169
        buf172 = empty((384, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_35.run(bitwise_or_13, buf170, buf172, 2688, 112, grid=grid(2688), stream=stream0)
        buf173 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_36.run(buf172, buf173, 384, 7, grid=grid(384), stream=stream0)
        buf174 = reinterpret_tensor(buf172, (384, 7), (1, 384), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_37.run(bitwise_or_13, buf170, convolution_31, primals_252, buf174, 2688, 112, grid=grid(2688), stream=stream0)
        del convolution_31
        del primals_252
        buf175 = empty((384, ), device='cuda', dtype=torch.float32)
        buf176 = buf175; del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_38.run(buf176, buf174, primals_253, 384, 7, grid=grid(384), stream=stream0)
        buf177 = empty_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_39.run(bitwise_or_13, buf170, primals_253, primals_95, buf177, 784, 384, grid=grid(784, 384), stream=stream0)
        del bitwise_or_13
        del buf170
        del primals_253
        del primals_95
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf178 = aten.convolution_backward(buf177, clamp_max_20, primals_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False])
        del clamp_max_20
        del primals_94
        buf179 = buf178[0]
        buf180 = buf178[1]
        del buf178
        buf181 = reinterpret_tensor(buf174, (384, 7), (7, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_35.run(bitwise_or_14, buf179, buf181, 2688, 112, grid=grid(2688), stream=stream0)
        buf182 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_36.run(buf181, buf182, 384, 7, grid=grid(384), stream=stream0)
        buf183 = reinterpret_tensor(buf181, (384, 7), (1, 384), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_37.run(bitwise_or_14, buf179, convolution_30, primals_249, buf183, 2688, 112, grid=grid(2688), stream=stream0)
        del convolution_30
        del primals_249
        buf184 = empty((384, ), device='cuda', dtype=torch.float32)
        buf185 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_38.run(buf185, buf183, primals_250, 384, 7, grid=grid(384), stream=stream0)
        buf186 = buf177; del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_39.run(bitwise_or_14, buf179, primals_250, primals_92, buf186, 784, 384, grid=grid(784, 384), stream=stream0)
        del bitwise_or_14
        del buf179
        del primals_250
        del primals_92
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf187 = aten.convolution_backward(buf186, add_65, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_65
        del primals_91
        buf188 = buf187[0]
        buf189 = buf187[1]
        del buf187
        buf190 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_40.run(buf188, buf190, 64, 784, grid=grid(64), stream=stream0)
        buf191 = empty((64, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_41.run(buf188, convolution_29, primals_246, buf191, 448, 112, grid=grid(448), stream=stream0)
        del convolution_29
        del primals_246
        buf192 = empty((64, ), device='cuda', dtype=torch.float32)
        buf193 = buf192; del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_42.run(buf193, buf191, primals_247, 64, 7, grid=grid(64), stream=stream0)
        del buf191
        buf194 = empty((4, 64, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_43.run(buf188, primals_247, primals_89, buf194, 50176, grid=grid(50176), stream=stream0)
        del primals_247
        del primals_89
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf195 = aten.convolution_backward(buf194, clamp_max_19, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clamp_max_19
        del primals_88
        buf196 = buf195[0]
        buf197 = buf195[1]
        del buf195
        buf198 = reinterpret_tensor(buf183, (384, 7), (7, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_35.run(bitwise_or_15, buf196, buf198, 2688, 112, grid=grid(2688), stream=stream0)
        buf199 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_36.run(buf198, buf199, 384, 7, grid=grid(384), stream=stream0)
        buf200 = reinterpret_tensor(buf198, (384, 7), (1, 384), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_37.run(bitwise_or_15, buf196, convolution_28, primals_243, buf200, 2688, 112, grid=grid(2688), stream=stream0)
        del convolution_28
        del primals_243
        buf201 = empty((384, ), device='cuda', dtype=torch.float32)
        buf202 = buf201; del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_38.run(buf202, buf200, primals_244, 384, 7, grid=grid(384), stream=stream0)
        buf203 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_39.run(bitwise_or_15, buf196, primals_244, primals_86, buf203, 784, 384, grid=grid(784, 384), stream=stream0)
        del bitwise_or_15
        del buf196
        del primals_244
        del primals_86
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf204 = aten.convolution_backward(buf203, clamp_max_18, primals_85, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False])
        del clamp_max_18
        del primals_85
        buf205 = buf204[0]
        buf206 = buf204[1]
        del buf204
        buf207 = reinterpret_tensor(buf200, (384, 7), (7, 1), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_35.run(bitwise_or_16, buf205, buf207, 2688, 112, grid=grid(2688), stream=stream0)
        buf208 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_36.run(buf207, buf208, 384, 7, grid=grid(384), stream=stream0)
        buf209 = reinterpret_tensor(buf207, (384, 7), (1, 384), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_37.run(bitwise_or_16, buf205, convolution_27, primals_240, buf209, 2688, 112, grid=grid(2688), stream=stream0)
        del convolution_27
        del primals_240
        buf210 = empty((384, ), device='cuda', dtype=torch.float32)
        buf211 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_38.run(buf211, buf209, primals_241, 384, 7, grid=grid(384), stream=stream0)
        buf212 = buf203; del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_39.run(bitwise_or_16, buf205, primals_241, primals_83, buf212, 784, 384, grid=grid(784, 384), stream=stream0)
        del bitwise_or_16
        del buf205
        del primals_241
        del primals_83
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf213 = aten.convolution_backward(buf212, add_58, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_58
        del primals_82
        buf214 = buf213[0]
        buf215 = buf213[1]
        del buf213
        buf219 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_44.run(buf188, buf214, primals_238, primals_80, buf219, 50176, grid=grid(50176), stream=stream0)
        del primals_80
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf220 = aten.convolution_backward(buf219, clamp_max_17, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clamp_max_17
        del primals_79
        buf221 = buf220[0]
        buf228 = buf212; del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_39.run(bitwise_or_17, buf221, primals_235, primals_77, buf228, 784, 384, grid=grid(784, 384), stream=stream0)
        del primals_77
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf229 = aten.convolution_backward(buf228, clamp_max_16, primals_76, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False])
        del clamp_max_16
        del primals_76
        buf230 = buf229[0]
        buf237 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_39.run(bitwise_or_18, buf230, primals_232, primals_74, buf237, 784, 384, grid=grid(784, 384), stream=stream0)
        del primals_74
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf238 = aten.convolution_backward(buf237, add_51, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_51
        del primals_73
        buf239 = buf238[0]
        buf244 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_45.run(buf188, buf214, buf239, primals_229, primals_71, buf244, 50176, grid=grid(50176), stream=stream0)
        del primals_71
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf245 = aten.convolution_backward(buf244, clamp_max_15, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf244
        del clamp_max_15
        del primals_70
        buf246 = buf245[0]
        buf253 = buf237; del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_39.run(bitwise_or_19, buf246, primals_226, primals_68, buf253, 784, 384, grid=grid(784, 384), stream=stream0)
        del primals_68
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf254 = aten.convolution_backward(buf253, clamp_max_14, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False])
        del clamp_max_14
        del primals_67
        buf255 = buf254[0]
        buf262 = buf253; del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_39.run(bitwise_or_20, buf255, primals_223, primals_65, buf262, 784, 384, grid=grid(784, 384), stream=stream0)
        del primals_65
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf263 = aten.convolution_backward(buf262, add_44, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_44
        del buf262
        del primals_64
        buf264 = buf263[0]
        buf216 = empty((64, ), device='cuda', dtype=torch.float32)
        buf217 = empty((64, ), device='cuda', dtype=torch.float32)
        buf241 = empty((64, ), device='cuda', dtype=torch.float32)
        buf242 = empty((64, ), device='cuda', dtype=torch.float32)
        buf266 = empty((64, ), device='cuda', dtype=torch.float32)
        buf267 = empty((64, ), device='cuda', dtype=torch.float32)
        buf218 = buf217; del buf217  # reuse
        buf243 = buf242; del buf242  # reuse
        buf268 = buf267; del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_46.run(buf218, buf243, buf268, buf188, buf214, convolution_26, primals_237, buf239, convolution_23, primals_228, buf264, convolution_20, primals_219, primals_238, primals_229, primals_220, buf216, buf241, buf266, 64, 784, grid=grid(64), stream=stream0)
        del convolution_20
        del convolution_23
        del convolution_26
        del primals_219
        del primals_228
        del primals_229
        del primals_237
        del primals_238
        buf222 = buf220[1]
        del buf220
        buf223 = reinterpret_tensor(buf209, (384, 7), (7, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_35.run(bitwise_or_17, buf221, buf223, 2688, 112, grid=grid(2688), stream=stream0)
        buf224 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_36.run(buf223, buf224, 384, 7, grid=grid(384), stream=stream0)
        buf225 = reinterpret_tensor(buf223, (384, 7), (1, 384), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_47.run(bitwise_or_17, buf221, convolution_25, primals_234, buf225, 2688, 112, grid=grid(2688), stream=stream0)
        del bitwise_or_17
        del buf221
        del convolution_25
        del primals_234
        buf226 = empty((384, ), device='cuda', dtype=torch.float32)
        buf227 = buf226; del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_38.run(buf227, buf225, primals_235, 384, 7, grid=grid(384), stream=stream0)
        del primals_235
        buf231 = buf229[1]
        del buf229
        buf232 = reinterpret_tensor(buf225, (384, 7), (7, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_35.run(bitwise_or_18, buf230, buf232, 2688, 112, grid=grid(2688), stream=stream0)
        buf233 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_36.run(buf232, buf233, 384, 7, grid=grid(384), stream=stream0)
        buf234 = reinterpret_tensor(buf232, (384, 7), (1, 384), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_47.run(bitwise_or_18, buf230, convolution_24, primals_231, buf234, 2688, 112, grid=grid(2688), stream=stream0)
        del bitwise_or_18
        del buf230
        del convolution_24
        del primals_231
        buf235 = empty((384, ), device='cuda', dtype=torch.float32)
        buf236 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_38.run(buf236, buf234, primals_232, 384, 7, grid=grid(384), stream=stream0)
        del primals_232
        buf240 = buf238[1]
        del buf238
        buf247 = buf245[1]
        del buf245
        buf248 = reinterpret_tensor(buf234, (384, 7), (7, 1), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_35.run(bitwise_or_19, buf246, buf248, 2688, 112, grid=grid(2688), stream=stream0)
        buf249 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_36.run(buf248, buf249, 384, 7, grid=grid(384), stream=stream0)
        buf250 = reinterpret_tensor(buf248, (384, 7), (1, 384), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_47.run(bitwise_or_19, buf246, convolution_22, primals_225, buf250, 2688, 112, grid=grid(2688), stream=stream0)
        del bitwise_or_19
        del buf246
        del convolution_22
        del primals_225
        buf251 = empty((384, ), device='cuda', dtype=torch.float32)
        buf252 = buf251; del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_38.run(buf252, buf250, primals_226, 384, 7, grid=grid(384), stream=stream0)
        del primals_226
        buf256 = buf254[1]
        del buf254
        buf257 = reinterpret_tensor(buf250, (384, 7), (7, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_35.run(bitwise_or_20, buf255, buf257, 2688, 112, grid=grid(2688), stream=stream0)
        buf258 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_36.run(buf257, buf258, 384, 7, grid=grid(384), stream=stream0)
        buf259 = reinterpret_tensor(buf257, (384, 7), (1, 384), 0); del buf257  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_47.run(bitwise_or_20, buf255, convolution_21, primals_222, buf259, 2688, 112, grid=grid(2688), stream=stream0)
        del bitwise_or_20
        del convolution_21
        del primals_222
        buf260 = empty((384, ), device='cuda', dtype=torch.float32)
        buf261 = buf260; del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_38.run(buf261, buf259, primals_223, 384, 7, grid=grid(384), stream=stream0)
        del buf259
        del primals_223
        buf265 = buf263[1]
        del buf263
        buf269 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_48.run(buf269, buf214, buf239, buf264, primals_220, primals_62, 50176, grid=grid(50176), stream=stream0)
        del buf214
        del buf239
        del buf264
        del primals_220
        del primals_62
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf270 = aten.convolution_backward(buf269, clamp_max_13, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf269
        del clamp_max_13
        del primals_61
        buf271 = buf270[0]
        buf272 = buf270[1]
        del buf270
        buf273 = empty((192, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_49.run(bitwise_or_21, buf271, buf273, 1344, 112, grid=grid(1344), stream=stream0)
        buf274 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_50.run(buf273, buf274, 192, 7, grid=grid(192), stream=stream0)
        buf275 = reinterpret_tensor(buf273, (192, 7), (1, 192), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_51.run(bitwise_or_21, buf271, convolution_19, primals_216, buf275, 1344, 112, grid=grid(1344), stream=stream0)
        del convolution_19
        del primals_216
        buf276 = empty((192, ), device='cuda', dtype=torch.float32)
        buf277 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_52.run(buf277, buf275, primals_217, 192, 7, grid=grid(192), stream=stream0)
        del buf275
        buf278 = empty_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_53.run(bitwise_or_21, buf271, primals_217, primals_59, buf278, 784, 192, grid=grid(784, 192), stream=stream0)
        del bitwise_or_21
        del buf271
        del primals_217
        del primals_59
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf279 = aten.convolution_backward(buf278, clamp_max_12, primals_58, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False])
        del buf278
        del clamp_max_12
        del primals_58
        buf280 = buf279[0]
        buf281 = buf279[1]
        del buf279
        buf282 = empty((192, 25), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_54.run(bitwise_or_22, buf280, buf282, 4800, 126, grid=grid(4800), stream=stream0)
        buf283 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_55.run(buf282, buf283, 192, 25, grid=grid(192), stream=stream0)
        buf284 = reinterpret_tensor(buf282, (192, 25), (1, 192), 0); del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_56.run(bitwise_or_22, buf280, convolution_18, primals_213, buf284, 4800, 126, grid=grid(4800), stream=stream0)
        del convolution_18
        del primals_213
        buf285 = empty((192, ), device='cuda', dtype=torch.float32)
        buf286 = buf285; del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_57.run(buf286, buf284, primals_214, 192, 25, grid=grid(192), stream=stream0)
        buf287 = empty_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_58.run(bitwise_or_22, buf280, primals_214, primals_56, buf287, 3136, 192, grid=grid(3136, 192), stream=stream0)
        del bitwise_or_22
        del buf280
        del primals_214
        del primals_56
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf288 = aten.convolution_backward(buf287, add_38, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_38
        del primals_55
        buf289 = buf288[0]
        buf290 = buf288[1]
        del buf288
        buf295 = empty((4, 32, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_59.run(buf289, primals_211, primals_53, buf295, 100352, grid=grid(100352), stream=stream0)
        del primals_53
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf296 = aten.convolution_backward(buf295, clamp_max_11, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clamp_max_11
        del primals_52
        buf297 = buf296[0]
        buf304 = buf287; del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_58.run(bitwise_or_23, buf297, primals_208, primals_50, buf304, 3136, 192, grid=grid(3136, 192), stream=stream0)
        del primals_50
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf305 = aten.convolution_backward(buf304, clamp_max_10, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False])
        del clamp_max_10
        del primals_49
        buf306 = buf305[0]
        buf313 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_58.run(bitwise_or_24, buf306, primals_205, primals_47, buf313, 3136, 192, grid=grid(3136, 192), stream=stream0)
        del primals_47
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf314 = aten.convolution_backward(buf313, add_31, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_31
        del primals_46
        buf315 = buf314[0]
        buf320 = buf295; del buf295  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_60.run(buf289, buf315, primals_202, primals_44, buf320, 100352, grid=grid(100352), stream=stream0)
        del primals_44
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf321 = aten.convolution_backward(buf320, clamp_max_9, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf320
        del clamp_max_9
        del primals_43
        buf322 = buf321[0]
        buf329 = buf313; del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_58.run(bitwise_or_25, buf322, primals_199, primals_41, buf329, 3136, 192, grid=grid(3136, 192), stream=stream0)
        del primals_41
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf330 = aten.convolution_backward(buf329, clamp_max_8, primals_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False])
        del clamp_max_8
        del primals_40
        buf331 = buf330[0]
        buf338 = buf329; del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_58.run(bitwise_or_26, buf331, primals_196, primals_38, buf338, 3136, 192, grid=grid(3136, 192), stream=stream0)
        del primals_38
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf339 = aten.convolution_backward(buf338, add_24, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_24
        del buf338
        del primals_37
        buf340 = buf339[0]
        buf291 = empty((32, ), device='cuda', dtype=torch.float32)
        buf317 = empty((32, ), device='cuda', dtype=torch.float32)
        buf318 = empty((32, ), device='cuda', dtype=torch.float32)
        buf342 = empty((32, ), device='cuda', dtype=torch.float32)
        buf343 = empty((32, ), device='cuda', dtype=torch.float32)
        buf319 = buf318; del buf318  # reuse
        buf344 = buf343; del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_61.run(buf319, buf344, buf289, buf315, convolution_14, primals_201, buf340, convolution_11, primals_192, primals_202, primals_193, buf291, buf317, buf342, 32, 3136, grid=grid(32), stream=stream0)
        del convolution_11
        del convolution_14
        del primals_192
        del primals_201
        del primals_202
        buf292 = empty((32, 25), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_62.run(buf289, convolution_17, primals_210, buf292, 800, 126, grid=grid(800), stream=stream0)
        del convolution_17
        del primals_210
        buf293 = empty((32, ), device='cuda', dtype=torch.float32)
        buf294 = buf293; del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_63.run(buf294, buf292, primals_211, 32, 25, grid=grid(32), stream=stream0)
        del buf292
        del primals_211
        buf298 = buf296[1]
        del buf296
        buf299 = reinterpret_tensor(buf284, (192, 25), (25, 1), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_54.run(bitwise_or_23, buf297, buf299, 4800, 126, grid=grid(4800), stream=stream0)
        buf300 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_55.run(buf299, buf300, 192, 25, grid=grid(192), stream=stream0)
        buf301 = reinterpret_tensor(buf299, (192, 25), (1, 192), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_64.run(bitwise_or_23, buf297, convolution_16, primals_207, buf301, 4800, 126, grid=grid(4800), stream=stream0)
        del bitwise_or_23
        del buf297
        del convolution_16
        del primals_207
        buf302 = empty((192, ), device='cuda', dtype=torch.float32)
        buf303 = buf302; del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_57.run(buf303, buf301, primals_208, 192, 25, grid=grid(192), stream=stream0)
        del primals_208
        buf307 = buf305[1]
        del buf305
        buf308 = reinterpret_tensor(buf301, (192, 25), (25, 1), 0); del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_54.run(bitwise_or_24, buf306, buf308, 4800, 126, grid=grid(4800), stream=stream0)
        buf309 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_55.run(buf308, buf309, 192, 25, grid=grid(192), stream=stream0)
        buf310 = reinterpret_tensor(buf308, (192, 25), (1, 192), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_64.run(bitwise_or_24, buf306, convolution_15, primals_204, buf310, 4800, 126, grid=grid(4800), stream=stream0)
        del bitwise_or_24
        del buf306
        del convolution_15
        del primals_204
        buf311 = empty((192, ), device='cuda', dtype=torch.float32)
        buf312 = buf311; del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_57.run(buf312, buf310, primals_205, 192, 25, grid=grid(192), stream=stream0)
        del primals_205
        buf316 = buf314[1]
        del buf314
        buf323 = buf321[1]
        del buf321
        buf324 = reinterpret_tensor(buf310, (192, 25), (25, 1), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_54.run(bitwise_or_25, buf322, buf324, 4800, 126, grid=grid(4800), stream=stream0)
        buf325 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_55.run(buf324, buf325, 192, 25, grid=grid(192), stream=stream0)
        buf326 = reinterpret_tensor(buf324, (192, 25), (1, 192), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_64.run(bitwise_or_25, buf322, convolution_13, primals_198, buf326, 4800, 126, grid=grid(4800), stream=stream0)
        del bitwise_or_25
        del buf322
        del convolution_13
        del primals_198
        buf327 = empty((192, ), device='cuda', dtype=torch.float32)
        buf328 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_57.run(buf328, buf326, primals_199, 192, 25, grid=grid(192), stream=stream0)
        del primals_199
        buf332 = buf330[1]
        del buf330
        buf333 = reinterpret_tensor(buf326, (192, 25), (25, 1), 0); del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_54.run(bitwise_or_26, buf331, buf333, 4800, 126, grid=grid(4800), stream=stream0)
        buf334 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_55.run(buf333, buf334, 192, 25, grid=grid(192), stream=stream0)
        buf335 = reinterpret_tensor(buf333, (192, 25), (1, 192), 0); del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_64.run(bitwise_or_26, buf331, convolution_12, primals_195, buf335, 4800, 126, grid=grid(4800), stream=stream0)
        del bitwise_or_26
        del buf331
        del convolution_12
        del primals_195
        buf336 = empty((192, ), device='cuda', dtype=torch.float32)
        buf337 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_57.run(buf337, buf335, primals_196, 192, 25, grid=grid(192), stream=stream0)
        del buf335
        del primals_196
        buf341 = buf339[1]
        del buf339
        buf345 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_65.run(buf345, buf315, buf340, primals_193, primals_35, 100352, grid=grid(100352), stream=stream0)
        del buf315
        del buf340
        del primals_193
        del primals_35
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf346 = aten.convolution_backward(buf345, clamp_max_7, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf345
        del clamp_max_7
        del primals_34
        buf347 = buf346[0]
        buf348 = buf346[1]
        del buf346
        buf349 = empty((144, 25), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_66.run(bitwise_or_27, buf347, buf349, 3600, 126, grid=grid(3600), stream=stream0)
        buf350 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_67.run(buf349, buf350, 144, 25, grid=grid(144), stream=stream0)
        buf351 = reinterpret_tensor(buf349, (144, 25), (1, 144), 0); del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_68.run(bitwise_or_27, buf347, convolution_10, primals_189, buf351, 3600, 126, grid=grid(3600), stream=stream0)
        del convolution_10
        del primals_189
        buf352 = empty((144, ), device='cuda', dtype=torch.float32)
        buf353 = buf352; del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_69.run(buf353, buf351, primals_190, 144, 25, grid=grid(144), stream=stream0)
        del buf351
        buf354 = reinterpret_tensor(buf154, (4, 144, 28, 28), (112896, 1, 4032, 144), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_70.run(bitwise_or_27, buf347, primals_190, primals_32, buf354, 3136, 144, grid=grid(3136, 144), stream=stream0)
        del bitwise_or_27
        del buf347
        del primals_190
        del primals_32
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf355 = aten.convolution_backward(buf354, clamp_max_6, primals_31, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False])
        del buf354
        del clamp_max_6
        del primals_31
        buf356 = buf355[0]
        buf357 = buf355[1]
        del buf355
        buf358 = empty((144, 98), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_71.run(bitwise_or_28, buf356, buf358, 14112, 128, grid=grid(14112), stream=stream0)
        buf359 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_72.run(buf358, buf359, 144, 98, grid=grid(144), stream=stream0)
        buf360 = reinterpret_tensor(buf358, (144, 98), (1, 144), 0); del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_73.run(bitwise_or_28, buf356, convolution_9, primals_186, buf360, 14112, 128, grid=grid(14112), stream=stream0)
        del convolution_9
        del primals_186
        buf361 = empty((144, ), device='cuda', dtype=torch.float32)
        buf362 = buf361; del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_74.run(buf362, buf360, primals_187, 144, 98, grid=grid(144), stream=stream0)
        buf363 = empty_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_75.run(bitwise_or_28, buf356, primals_187, primals_29, buf363, 12544, 144, grid=grid(12544, 144), stream=stream0)
        del bitwise_or_28
        del buf356
        del primals_187
        del primals_29
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf364 = aten.convolution_backward(buf363, add_18, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_18
        del primals_28
        buf365 = buf364[0]
        buf366 = buf364[1]
        del buf364
        buf372 = reinterpret_tensor(buf255, (4, 24, 56, 56), (75264, 3136, 56, 1), 0); del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_76.run(buf365, primals_184, primals_26, buf372, 301056, grid=grid(301056), stream=stream0)
        del primals_26
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf373 = aten.convolution_backward(buf372, clamp_max_5, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf372
        del clamp_max_5
        del primals_25
        buf374 = buf373[0]
        buf381 = buf363; del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_75.run(bitwise_or_29, buf374, primals_181, primals_23, buf381, 12544, 144, grid=grid(12544, 144), stream=stream0)
        del primals_23
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf382 = aten.convolution_backward(buf381, clamp_max_4, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False])
        del clamp_max_4
        del primals_22
        buf383 = buf382[0]
        buf390 = buf381; del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_75.run(bitwise_or_30, buf383, primals_178, primals_20, buf390, 12544, 144, grid=grid(12544, 144), stream=stream0)
        del primals_20
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf391 = aten.convolution_backward(buf390, add_11, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_11
        del buf390
        del primals_19
        buf392 = buf391[0]
        buf367 = empty_strided((24, 2), (1, 24), device='cuda', dtype=torch.float32)
        buf394 = empty_strided((24, 2), (1, 24), device='cuda', dtype=torch.float32)
        buf396 = empty_strided((24, 2), (1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_77.run(buf365, buf392, convolution_5, primals_174, buf367, buf394, buf396, 48, 6272, grid=grid(48), stream=stream0)
        del convolution_5
        del primals_174
        buf368 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_78.run(buf367, buf368, 24, 2, grid=grid(24), stream=stream0)
        del buf367
        buf369 = empty((24, 98), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_79.run(buf365, convolution_8, primals_183, buf369, 2352, 128, grid=grid(2352), stream=stream0)
        del convolution_8
        del primals_183
        buf370 = empty((24, ), device='cuda', dtype=torch.float32)
        buf371 = buf370; del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_80.run(buf371, buf369, primals_184, 24, 98, grid=grid(24), stream=stream0)
        del buf369
        del primals_184
        buf375 = buf373[1]
        del buf373
        buf376 = reinterpret_tensor(buf360, (144, 98), (98, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_71.run(bitwise_or_29, buf374, buf376, 14112, 128, grid=grid(14112), stream=stream0)
        buf377 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_72.run(buf376, buf377, 144, 98, grid=grid(144), stream=stream0)
        buf378 = reinterpret_tensor(buf376, (144, 98), (1, 144), 0); del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_81.run(bitwise_or_29, buf374, convolution_7, primals_180, buf378, 14112, 128, grid=grid(14112), stream=stream0)
        del bitwise_or_29
        del buf374
        del convolution_7
        del primals_180
        buf379 = empty((144, ), device='cuda', dtype=torch.float32)
        buf380 = buf379; del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_74.run(buf380, buf378, primals_181, 144, 98, grid=grid(144), stream=stream0)
        del primals_181
        buf384 = buf382[1]
        del buf382
        buf385 = reinterpret_tensor(buf378, (144, 98), (98, 1), 0); del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_71.run(bitwise_or_30, buf383, buf385, 14112, 128, grid=grid(14112), stream=stream0)
        buf386 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_72.run(buf385, buf386, 144, 98, grid=grid(144), stream=stream0)
        buf387 = reinterpret_tensor(buf385, (144, 98), (1, 144), 0); del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_81.run(bitwise_or_30, buf383, convolution_6, primals_177, buf387, 14112, 128, grid=grid(14112), stream=stream0)
        del bitwise_or_30
        del buf383
        del convolution_6
        del primals_177
        buf388 = empty((144, ), device='cuda', dtype=torch.float32)
        buf389 = buf388; del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_74.run(buf389, buf387, primals_178, 144, 98, grid=grid(144), stream=stream0)
        del buf387
        del primals_178
        buf393 = buf391[1]
        del buf391
        buf395 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_78.run(buf394, buf395, 24, 2, grid=grid(24), stream=stream0)
        del buf394
        buf397 = empty((24, ), device='cuda', dtype=torch.float32)
        buf398 = buf397; del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_82.run(buf398, buf396, primals_175, 24, 2, grid=grid(24), stream=stream0)
        del buf396
        buf399 = buf365; del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_83.run(buf399, buf392, primals_175, primals_17, 301056, grid=grid(301056), stream=stream0)
        del buf392
        del primals_17
        del primals_175
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf400 = aten.convolution_backward(buf399, clamp_max_3, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf399
        del clamp_max_3
        del primals_16
        buf401 = buf400[0]
        buf402 = buf400[1]
        del buf400
        buf403 = empty((96, 98), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_84.run(bitwise_or_31, buf401, buf403, 9408, 128, grid=grid(9408), stream=stream0)
        buf404 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_85.run(buf403, buf404, 96, 98, grid=grid(96), stream=stream0)
        buf405 = reinterpret_tensor(buf403, (96, 98), (1, 96), 0); del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_86.run(bitwise_or_31, buf401, convolution_4, primals_171, buf405, 9408, 128, grid=grid(9408), stream=stream0)
        del convolution_4
        del primals_171
        buf406 = empty((96, ), device='cuda', dtype=torch.float32)
        buf407 = buf406; del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_87.run(buf407, buf405, primals_172, 96, 98, grid=grid(96), stream=stream0)
        del buf405
        buf408 = empty_strided((4, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_88.run(bitwise_or_31, buf401, primals_172, primals_14, buf408, 12544, 96, grid=grid(12544, 96), stream=stream0)
        del bitwise_or_31
        del buf401
        del primals_14
        del primals_172
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf409 = aten.convolution_backward(buf408, clamp_max_2, primals_13, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 96, [True, True, False])
        del buf408
        del clamp_max_2
        del primals_13
        buf410 = buf409[0]
        buf411 = buf409[1]
        del buf409
        buf412 = empty((96, 392), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_89.run(bitwise_or_32, buf410, buf412, 37632, 128, grid=grid(37632), stream=stream0)
        buf413 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_90.run(buf412, buf413, 96, 392, grid=grid(96), stream=stream0)
        buf414 = reinterpret_tensor(buf412, (96, 392), (1, 96), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_91.run(bitwise_or_32, buf410, convolution_3, primals_168, buf414, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_3
        del primals_168
        buf415 = empty((96, ), device='cuda', dtype=torch.float32)
        buf416 = buf415; del buf415  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_92.run(buf416, buf414, primals_169, 96, 392, grid=grid(96), stream=stream0)
        del buf414
        buf417 = empty_strided((4, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_93.run(bitwise_or_32, buf410, primals_169, primals_11, buf417, 50176, 96, grid=grid(50176, 96), stream=stream0)
        del bitwise_or_32
        del buf410
        del primals_11
        del primals_169
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf418 = aten.convolution_backward(buf417, add_5, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_5
        del buf417
        del primals_10
        buf419 = buf418[0]
        buf420 = buf418[1]
        del buf418
        buf421 = empty((16, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_94.run(buf419, buf421, 112, 7168, grid=grid(112), stream=stream0)
        buf422 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_95.run(buf421, buf422, 16, 7, grid=grid(16), stream=stream0)
        del buf421
        buf423 = empty((16, 392), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_96.run(buf419, convolution_2, primals_165, buf423, 6272, 128, grid=grid(6272), stream=stream0)
        del convolution_2
        del primals_165
        buf424 = empty((16, ), device='cuda', dtype=torch.float32)
        buf425 = buf424; del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_97.run(buf425, buf423, primals_166, 16, 392, grid=grid(16), stream=stream0)
        del buf423
        buf426 = buf419; del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_98.run(buf426, primals_166, primals_8, 802816, grid=grid(802816), stream=stream0)
        del primals_166
        del primals_8
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf427 = aten.convolution_backward(buf426, clamp_max_1, primals_7, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf426
        del clamp_max_1
        del primals_7
        buf428 = buf427[0]
        buf429 = buf427[1]
        del buf427
        buf430 = empty((32, 392), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_99.run(bitwise_or_33, buf428, buf430, 12544, 128, grid=grid(12544), stream=stream0)
        buf431 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_100.run(buf430, buf431, 32, 392, grid=grid(32), stream=stream0)
        buf432 = reinterpret_tensor(buf430, (32, 392), (1, 32), 0); del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_101.run(bitwise_or_33, buf428, convolution_1, primals_162, buf432, 12544, 128, grid=grid(12544), stream=stream0)
        del convolution_1
        del primals_162
        buf433 = empty((32, ), device='cuda', dtype=torch.float32)
        buf434 = buf433; del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_102.run(buf434, buf432, primals_163, 32, 392, grid=grid(32), stream=stream0)
        buf435 = empty_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_103.run(bitwise_or_33, buf428, primals_163, primals_5, buf435, 50176, 32, grid=grid(50176, 32), stream=stream0)
        del bitwise_or_33
        del buf428
        del primals_163
        del primals_5
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf436 = aten.convolution_backward(buf435, clamp_max, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del clamp_max
        del primals_4
        buf437 = buf436[0]
        buf438 = buf436[1]
        del buf436
        buf439 = reinterpret_tensor(buf432, (32, 392), (392, 1), 0); del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_99.run(bitwise_or_34, buf437, buf439, 12544, 128, grid=grid(12544), stream=stream0)
        buf440 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_100.run(buf439, buf440, 32, 392, grid=grid(32), stream=stream0)
        buf441 = reinterpret_tensor(buf439, (32, 392), (1, 32), 0); del buf439  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_101.run(bitwise_or_34, buf437, convolution, primals_159, buf441, 12544, 128, grid=grid(12544), stream=stream0)
        del convolution
        del primals_159
        buf442 = empty((32, ), device='cuda', dtype=torch.float32)
        buf443 = buf442; del buf442  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_102.run(buf443, buf441, primals_160, 32, 392, grid=grid(32), stream=stream0)
        del buf441
        buf444 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_103.run(bitwise_or_34, buf437, primals_160, primals_2, buf444, 50176, 32, grid=grid(50176, 32), stream=stream0)
        del bitwise_or_34
        del buf437
        del primals_160
        del primals_2
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf445 = aten.convolution_backward(buf444, primals_315, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf444
        del primals_1
        del primals_315
        buf446 = buf445[1]
        return (buf446, buf443, buf440, buf438, buf434, buf431, buf429, buf425, buf422, buf420, buf416, buf413, buf411, buf407, buf404, buf402, buf398, buf395, buf393, buf389, buf386, buf384, buf380, buf377, buf375, buf371, buf368, buf366, buf362, buf359, buf357, buf353, buf350, buf348, buf344, buf342, buf341, buf337, buf334, buf332, buf328, buf325, buf323, buf319, buf317, buf316, buf312, buf309, buf307, buf303, buf300, buf298, buf294, buf291, buf290, buf286, buf283, buf281, buf277, buf274, buf272, buf268, buf266, buf265, buf261, buf258, buf256, buf252, buf249, buf247, buf243, buf241, buf240, buf236, buf233, buf231, buf227, buf224, buf222, buf218, buf216, buf215, buf211, buf208, buf206, buf202, buf199, buf197, buf193, buf190, buf189, buf185, buf182, buf180, buf176, buf173, buf171, buf167, buf165, buf164, buf160, buf157, buf155, buf151, buf148, buf146, buf142, buf140, buf139, buf135, buf132, buf130, buf126, buf123, buf121, buf117, buf114, buf113, buf109, buf106, buf104, buf100, buf97, buf95, buf91, buf89, buf88, buf84, buf81, buf79, buf75, buf72, buf70, buf66, buf64, buf63, buf59, buf56, buf54, buf50, buf47, buf45, buf41, buf38, buf37, buf33, buf30, buf28, buf24, buf21, buf19, buf15, buf12, buf11, buf7, buf4, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), buf2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((32, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((160, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((320, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    clamp_max = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    clamp_max_1 = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    add_5 = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((4, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda:0', dtype=torch.float32)
    clamp_max_2 = rand_strided((4, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((4, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    clamp_max_3 = rand_strided((4, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    add_11 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    clamp_max_4 = rand_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    clamp_max_5 = rand_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    add_18 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    clamp_max_6 = rand_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((4, 144, 28, 28), (112896, 1, 4032, 144), device='cuda:0', dtype=torch.float32)
    clamp_max_7 = rand_strided((4, 144, 28, 28), (112896, 1, 4032, 144), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((4, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    add_24 = rand_strided((4, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    clamp_max_8 = rand_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    clamp_max_9 = rand_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((4, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    add_31 = rand_strided((4, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    clamp_max_10 = rand_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    clamp_max_11 = rand_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((4, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    add_38 = rand_strided((4, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    clamp_max_12 = rand_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cuda:0', dtype=torch.float32)
    clamp_max_13 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((4, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    add_44 = rand_strided((4, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    clamp_max_14 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    clamp_max_15 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((4, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    add_51 = rand_strided((4, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    clamp_max_16 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    clamp_max_17 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((4, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    add_58 = rand_strided((4, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    clamp_max_18 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    clamp_max_19 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((4, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    add_65 = rand_strided((4, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    clamp_max_20 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    clamp_max_21 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cuda:0', dtype=torch.float32)
    add_71 = rand_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    clamp_max_22 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    clamp_max_23 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cuda:0', dtype=torch.float32)
    add_78 = rand_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    clamp_max_24 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    clamp_max_25 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cuda:0', dtype=torch.float32)
    add_85 = rand_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    clamp_max_26 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((4, 576, 7, 7), (28224, 1, 4032, 576), device='cuda:0', dtype=torch.float32)
    clamp_max_27 = rand_strided((4, 576, 7, 7), (28224, 1, 4032, 576), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    add_91 = rand_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    clamp_max_28 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    clamp_max_29 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    add_98 = rand_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    clamp_max_30 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    clamp_max_31 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    add_105 = rand_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    clamp_max_32 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    clamp_max_33 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((4, 320, 7, 7), (15680, 1, 2240, 320), device='cuda:0', dtype=torch.float32)
    add_111 = rand_strided((4, 320, 7, 7), (15680, 1, 2240, 320), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda:0', dtype=torch.float32)
    clone_35 = rand_strided((4, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or = rand_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda:0', dtype=torch.bool)
    bitwise_or_1 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.bool)
    bitwise_or_2 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.bool)
    bitwise_or_3 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.bool)
    bitwise_or_4 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.bool)
    bitwise_or_5 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.bool)
    bitwise_or_6 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.bool)
    bitwise_or_7 = rand_strided((4, 576, 7, 7), (28224, 1, 4032, 576), device='cuda:0', dtype=torch.bool)
    bitwise_or_8 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.bool)
    bitwise_or_9 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.bool)
    bitwise_or_10 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.bool)
    bitwise_or_11 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.bool)
    bitwise_or_12 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.bool)
    bitwise_or_13 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.bool)
    bitwise_or_14 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.bool)
    bitwise_or_15 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.bool)
    bitwise_or_16 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.bool)
    bitwise_or_17 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.bool)
    bitwise_or_18 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.bool)
    bitwise_or_19 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.bool)
    bitwise_or_20 = rand_strided((4, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.bool)
    bitwise_or_21 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cuda:0', dtype=torch.bool)
    bitwise_or_22 = rand_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.bool)
    bitwise_or_23 = rand_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.bool)
    bitwise_or_24 = rand_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.bool)
    bitwise_or_25 = rand_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.bool)
    bitwise_or_26 = rand_strided((4, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.bool)
    bitwise_or_27 = rand_strided((4, 144, 28, 28), (112896, 1, 4032, 144), device='cuda:0', dtype=torch.bool)
    bitwise_or_28 = rand_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.bool)
    bitwise_or_29 = rand_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.bool)
    bitwise_or_30 = rand_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.bool)
    bitwise_or_31 = rand_strided((4, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.bool)
    bitwise_or_32 = rand_strided((4, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda:0', dtype=torch.bool)
    bitwise_or_33 = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.bool)
    bitwise_or_34 = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.bool)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_159, primals_160, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, convolution, clamp_max, convolution_1, clamp_max_1, convolution_2, add_5, convolution_3, clamp_max_2, convolution_4, clamp_max_3, convolution_5, add_11, convolution_6, clamp_max_4, convolution_7, clamp_max_5, convolution_8, add_18, convolution_9, clamp_max_6, convolution_10, clamp_max_7, convolution_11, add_24, convolution_12, clamp_max_8, convolution_13, clamp_max_9, convolution_14, add_31, convolution_15, clamp_max_10, convolution_16, clamp_max_11, convolution_17, add_38, convolution_18, clamp_max_12, convolution_19, clamp_max_13, convolution_20, add_44, convolution_21, clamp_max_14, convolution_22, clamp_max_15, convolution_23, add_51, convolution_24, clamp_max_16, convolution_25, clamp_max_17, convolution_26, add_58, convolution_27, clamp_max_18, convolution_28, clamp_max_19, convolution_29, add_65, convolution_30, clamp_max_20, convolution_31, clamp_max_21, convolution_32, add_71, convolution_33, clamp_max_22, convolution_34, clamp_max_23, convolution_35, add_78, convolution_36, clamp_max_24, convolution_37, clamp_max_25, convolution_38, add_85, convolution_39, clamp_max_26, convolution_40, clamp_max_27, convolution_41, add_91, convolution_42, clamp_max_28, convolution_43, clamp_max_29, convolution_44, add_98, convolution_45, clamp_max_30, convolution_46, clamp_max_31, convolution_47, add_105, convolution_48, clamp_max_32, convolution_49, clamp_max_33, convolution_50, add_111, convolution_51, clone_35, permute_1, bitwise_or, bitwise_or_1, bitwise_or_2, bitwise_or_3, bitwise_or_4, bitwise_or_5, bitwise_or_6, bitwise_or_7, bitwise_or_8, bitwise_or_9, bitwise_or_10, bitwise_or_11, bitwise_or_12, bitwise_or_13, bitwise_or_14, bitwise_or_15, bitwise_or_16, bitwise_or_17, bitwise_or_18, bitwise_or_19, bitwise_or_20, bitwise_or_21, bitwise_or_22, bitwise_or_23, bitwise_or_24, bitwise_or_25, bitwise_or_26, bitwise_or_27, bitwise_or_28, bitwise_or_29, bitwise_or_30, bitwise_or_31, bitwise_or_32, bitwise_or_33, bitwise_or_34, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilenet_v2', benchmark_compiled_module)
