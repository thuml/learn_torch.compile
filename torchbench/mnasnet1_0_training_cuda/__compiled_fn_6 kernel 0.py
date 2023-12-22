
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


# kernel path: /tmp/torchinductor_youkaichao/kh/ckhdlufoq2qszfwta6k5bg5knaksdgbpnzrm72r6f6vcvpxkswdm.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_div_native_batch_norm_backward_threshold_backward_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/5x/c5xvnrteze75hxoqec7ownzokd4py7r7unfxynhri3ipzm4nfkxp.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_div_native_batch_norm_backward_threshold_backward_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/nd/cndvkxpvedrunnogkndphduqyuahl4nadx4xnzl7cyyzjfxcjwdq.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_div_native_batch_norm_backward_threshold_backward_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/fs/cfsnz7eha3xxjky5t4is3yfbwanyautxuhk2lmbdfanhshkfsnjl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/no/cno4r2c2xfhjcmbolwoc7s7pv2jnhgw3ukb6v3vyvajepp2v5fqx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2304
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1152
    x1 = (xindex // 1152)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1152*r2) + (112896*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((49*x0) + (56448*(r2 // 49)) + (112896*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (1152*r2) + (112896*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ia/ciaongtkv4txjytc7z7r2qgsjwhzsv7tcsj23o7cb6f4rtdarjzl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1152*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2s/c2sz6t6j3kuj74qdlyhs6muhzx2mzl5a6tjrtofxseihsb2hb7mr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1152*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/2e/c2ekxjwslj24q54xd2w654ibkknmnffpnakhvrp2qmw46gvafvtg.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
    xnumel = 1152
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1152*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (49*x2) + (56448*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (1152*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h3/ch32fxyteoeg2n6aaatzmelopauq6n5ycbft4tlvh7lct3gfvl6y.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (9408*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l7/cl7rwoevca4447mtaugwgvnmsc2r35cdhajlb7exrcorggqxg6fk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (9408*(r2 // 49)) + (18816*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (192*r2) + (18816*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tk/ctkiu4mhbk5qy722u3kdmc5xd6nv7q3zaxyhgg54kjw42plfk6c3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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


# kernel path: /tmp/torchinductor_youkaichao/gk/cgkg2dwogcy6hxdqvor5drmftxa3vbmdnewzzz2zidplu6nesatj.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 192
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


# kernel path: /tmp/torchinductor_youkaichao/wf/cwfnjsre6ohdmojpjoy2ev5thiskcw4smguqzcjdrl7tpwpxkl3t.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 192
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


# kernel path: /tmp/torchinductor_youkaichao/35/c35kqnusgvapj62mk6foh3ojgimurid2yk5lwtusa3kfenkois44.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 192
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


# kernel path: /tmp/torchinductor_youkaichao/wu/cwuuh3lcdzpntoyhiiojxfp2oynouk4piig3hpuiivy7nlonulrx.py
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
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: 'i32', 20: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(19,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_19', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (9408*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (9408*r2)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0 + (192*r3)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (r1 + (49*x0) + (9408*r2)), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr5 + (x0 + (192*r3)), rmask & xmask, other=0.0)
    tmp22 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (r1 + (49*x0) + (9408*r2)), rmask & xmask, other=0.0)
    tmp35 = tl.load(in_ptr8 + (x0 + (192*r3)), rmask & xmask, other=0.0)
    tmp36 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr12 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp9 = tmp7 - tmp8
    tmp10 = tmp2 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp16 = tmp2 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp23 = tmp21 - tmp22
    tmp24 = tmp16 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None]
    tmp30 = tmp16 + tmp29
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tmp37 = tmp35 - tmp36
    tmp38 = tmp30 * tmp37
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
    tmp41 = tl.where(rmask & xmask, tmp39, 0)
    tmp42 = tl.sum(tmp41, 1)[:, None]
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


# kernel path: /tmp/torchinductor_youkaichao/ac/cacaylbgh4it7thmdoljsgpka4hj5nhfahi4rskprkjh747wbxvj.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 192
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_out_ptr0 + (x3), xmask)
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


# kernel path: /tmp/torchinductor_youkaichao/cg/ccgf3ijpqxmz5oe5t4swzbb5bcsidy6rba5q4pnjalozsiv6pqpd.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': []}
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
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (576*r2) + (56448*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((49*x0) + (28224*(r2 // 49)) + (56448*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (576*r2) + (56448*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ha/chafvru35liwpoxra2fubtv53ojy6q7hydlpac4qgli7r5adq7zm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_22', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/4l/c4lexk47p7g6kbykuty5k4z7ae27j7c3wqtocyxmfh6enmr2lauv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/or/corm36kyozaf4kt3r3kcr76yapahvnwviznywfo4udrtcherhfbl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (x2 + (576*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (49*x2) + (28224*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (576*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qu/cqujxqaqokfi73c634txkekcrnuybu52aldduy5htr5untpn3hkb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_25', 'mutated_arg_names': []}
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
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (576*r2) + (64512*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (112896*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7j/c7jixryeb55obxbcoiryfa2vecbqqiswdikuqc43xqacvsfzivho.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/le/clej3faxozlllag2mtbv6x2xgjuqdcecykemam6uhjzrksfdxf7u.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_27', 'mutated_arg_names': []}
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
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (576*r2) + (64512*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (112896*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (576*r2) + (64512*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/fd/cfdlngyxhn75hm2fca4u3j3czflsegeiiry7nys63ybfjv5q6fy3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/37/c37opnyl3kglishbec5qyost3bmsjo5v3s2yplhf7mm5nzjarp3o.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (x2 + (576*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (196*x2) + (112896*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (576*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7w/c7w4ficy7iosnuwvmoeu7fonrrsmmal5xczcrcyfkggknw6itves.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_30', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/pv/cpv4zjtk32yhkqtvwd63odw2konxz5rdgthoxbkx74hsdxyaafbx.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_31', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tmp19 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp18 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xx/cxxmtdhuzhuzudjkqieb3xx6bocz2r7ay6vrb3ozzjrnrz5vom7f.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_32', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/f6/cf6pbtw6rddq2frlietlxteaasrqjur2ew2yyqkasewsyyxnq4sw.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_33', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/ca/cca2vrzu6fuisdltrc2myzxgxf6epsch5z2pcpsfboepvtoa5fz7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_34', 'mutated_arg_names': []}
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
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (576*r2) + (64512*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (112896*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (576*r2) + (64512*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qw/cqwr3g4ni67pru3namqp532z4amfzlgndyynfhb5ndn7e3h5byn5.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_35', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 96
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.math.rsqrt(tmp5)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 * tmp8
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/se/csex43zbvihyra2chdivldqsyikufwphx3ne6qigrusb4nevlq4g.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3360
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (480*r2) + (53760*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (94080*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fv/cfvbchnefzmnz72mvbky24jn4nfo722ohxii24sd6nxrz7hvrqqs.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
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


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xdvk6snnhctvwbbkt7u5hz54epzhalbktqu745m4hvrw42wlx3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3360
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 480
    x1 = (xindex // 480)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (53760*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (94080*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (480*r2) + (53760*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/hi/chi7et3eufvs7edldahhxcoabeg6ghiphxvb6fvcarn53d3kbxkn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_39', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 7
    RBLOCK: tl.constexpr = 8
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lu/clu7lz4tfgsdpvg2ipxfswfujbbz753y7odfd2rc35g37rt544jw.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
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
    tmp3 = tl.load(in_ptr1 + (y0 + (196*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (480*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o3/co3r2rhmye7e5gg6numhxfcf7ellf6fqfyogdi7aq6vixscvdzsv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 80
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


# kernel path: /tmp/torchinductor_youkaichao/bu/cbu3mxrjbr56b524r7fpodqyawo7xy36s52tssgdrsr6ti2bgur4.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 80
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


# kernel path: /tmp/torchinductor_youkaichao/qj/cqj44tfyf7lgkjvkkumdstjlzs3uaws2erhg6erjgi7ovqhpcuqu.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_43', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 80
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (x0 + (80*r3)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr5 + (x0 + (80*r3)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/m6/cm6voa6p2xl3tecnqkxqm4jarmdlrbfuffevaav3lj5gtyzvecot.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 560
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
        tmp0 = tl.load(in_ptr0 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (15680*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (80*r2) + (8960*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nk/cnkephtz7dwlg5apmhzotvv5z3wu6feykvgjroj3gjthtewwlec7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_45', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 80
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


# kernel path: /tmp/torchinductor_youkaichao/5r/c5rwsekci5yxh3tp6p7alrpy6xg5df4bsvgimndlx43x654tmzx5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3360
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 480
    x1 = (xindex // 480)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (53760*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (94080*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (480*r2) + (53760*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qk/cqkg3ivg72cgj65dfckmmlnb266bfp4ai5wkxo3cjecqb3f3yazw.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 80
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


# kernel path: /tmp/torchinductor_youkaichao/ul/culsqkcif57ce52isbvub7w2mrpcnxf5cywrd7kuzfvq4lv7exgb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1680
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (240*r2) + (26880*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (47040*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3r/c3rrir6exaag6ymoavuhhigubgylkmihsqwor2qlrsbbii5do6le.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
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


# kernel path: /tmp/torchinductor_youkaichao/ql/cqlrvgapfdcnecv4pt7wn2gnw3ygmjwofaal3c5oulwavczx63wi.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1680
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 240
    x1 = (xindex // 240)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (26880*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (47040*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (240*r2) + (26880*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/kj/ckjg4islh6lba4azsax6sgruwu6xj3pwimfdyrsu353nh5phdz2r.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_51', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 7
    RBLOCK: tl.constexpr = 8
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z4/cz4c3r7oc4rtdkxtyyi2fr75eai27v63jjvr7u6lyd37r3judrh5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
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
    tmp3 = tl.load(in_ptr1 + (y0 + (196*x2) + (47040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (240*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i7/ci7fptt72ecnot5hpyzvyl72cuugufc2dqaqswuokrkisyy5h6qp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6000
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
        tmp3 = tl.load(in_ptr0 + (x1 + (240*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((784*x1) + (188160*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hf/chf3hj2fbilnplvysxh4o2xrv5kpz2hsc4wu4pfpqan2mhxivqgm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
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


# kernel path: /tmp/torchinductor_youkaichao/pe/cpeqmu3miibbze7n5zzqzuee4ydio5a5w3v7wfi6rg7kwa7jjce6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6000
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 240)
    x0 = xindex % 240
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (240*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((784*x0) + (188160*(((r2 + (126*x1)) // 784) % 4)) + ((r2 + (126*x1)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (240*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/sc/cscvmsecnoy4wq2vbaghutowclhwqc4iw5whievaqam6lmfpammt.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_56', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 25
    RBLOCK: tl.constexpr = 32
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ke/ckeds2pq6r2xa45mlw37e2mxmsqbr7pafk4aiahplka2esvkq46z.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
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
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (240*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ur/cur6xzmsfprk6kobwxlthea2bltlkri4sgghsczeghvku25j4g5p.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 40
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


# kernel path: /tmp/torchinductor_youkaichao/az/cazznti2sn52nomovz7ubrwbjbzttdfq3edp472ezu6quoyozuxp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
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
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (120*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vd/cvdrwlslb66pdhtdlcemuo5n2ybklwbui5wq4jtxpgoz6tilk2iv.py
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
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 40
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


# kernel path: /tmp/torchinductor_youkaichao/df/cdf5gzny6rbqa7w6u6pmsczhj5vvd7jofcsyjnc7fmqosgum2zj6.py
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
    size_hints=[64, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_61', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 40
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (784*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (40*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr4 + (r1 + (784*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr5 + (x0 + (40*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/y5/cy5y4f4bhrpxegsz7xyrwrgduqsmugax3zc2pg4q4aoz52yejmpg.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1000
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
        tmp3 = tl.load(in_ptr0 + ((784*x1) + (31360*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (40*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/76/c76g5x52j3lgmdg4mosmw4myc5dbacpbwkjbh2wudx23ekm6vrqa.py
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
    size_hints=[64, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_63', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 40
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


# kernel path: /tmp/torchinductor_youkaichao/cy/ccylortpw3g3abm5gdrvkz3mvd3lrngfwddxisdte6vh7bmdxj4c.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3000
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
        tmp3 = tl.load(in_ptr0 + (x1 + (120*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((784*x1) + (94080*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/65/c65kmrwjg6jfn2enwv5hdoahlknszdtzgx6bvsyozxe7jmb3iilz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 120
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


# kernel path: /tmp/torchinductor_youkaichao/sc/cscmjrcflvqp7x6sdhkigrdgrq7n7r6mh3v22f2qs4psnu4t3uwu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3000
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 120)
    x0 = xindex % 120
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (120*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((784*x0) + (94080*(((r2 + (126*x1)) // 784) % 4)) + ((r2 + (126*x1)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (120*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/wz/cwzzaum53botjwiq5mzocn6n6cpr5romi52tdf4237xmsa27nio5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_67', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 120
    rnumel = 25
    RBLOCK: tl.constexpr = 32
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mc/cmcfwghqxdolcmtfs3ikcxhhlnctzprxlh5dbe5rdsgnyrb2g3c4.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_68', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 40
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


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5uauhqdoxt26w7zfzr4ibvhbdygljzpmfxva5mhww33cxqe2e3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1800
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
        tmp3 = tl.load(in_ptr0 + (x1 + (72*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((784*x1) + (56448*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ri/crix62gp5iqjie7ryssy6efbxaaqbn5e3na6gzgnhzzbm2v5tkx7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_70 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 72
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


# kernel path: /tmp/torchinductor_youkaichao/cz/cczc46yyx2zsuznp7tsz6wb7h5z36g75ukywbjafpobdmhadfzxy.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 72)
    x0 = xindex % 72
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (72*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((784*x0) + (56448*(((r2 + (126*x1)) // 784) % 4)) + ((r2 + (126*x1)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (72*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4r/c4rqdqbf6cnb7jqv5nhwock6vushdjlaldyk4nf6l7rrm622xoz4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_72', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 72
    rnumel = 25
    RBLOCK: tl.constexpr = 32
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sq/csq775u5jllw5f45ttmf3mvszngog2krffrfxymq6hotwwgnlpcc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_73 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
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
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (72*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/63/c63etsn2wcinfkm533jqzhgnhmh6kbqrbpvuwnzede42hefgbpla.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7056
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 98
    x1 = (xindex // 98)
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


# kernel path: /tmp/torchinductor_youkaichao/gr/cgrrzbxh5o67oxwf444cub6i24z4tvctbuekbjw2gxnpfi4vyqed.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 72
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


# kernel path: /tmp/torchinductor_youkaichao/5b/c5bvg6zz24mcapnqfd6ikgwmcdmhdg3iwkonrw67sbeylfieftih.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7056
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
        tmp5 = tl.load(in_ptr2 + (x0 + (72*r2) + (9216*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/vl/cvl7z2qcmrx5p333t5jjwforfcenftvt3z7fkja6i2hmplynsdnn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_77', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
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
        tmp0 = tl.load(in_ptr0 + (x0 + (72*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bx/cbxvkgefkgm3ad56xmtflxszv56gopuclesoooo2m5dhpx5dqh2p.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
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
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (72*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qy/cqy4u4sjiuplobfrpwn7adtixapyp3wv4nsw7aimec5vfjkx3bo7.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_79', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/jm/cjmuixz33w34mil4bltk4t3qaqkyot7ota65k4ecb6ecrvhzulif.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 24
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


# kernel path: /tmp/torchinductor_youkaichao/ys/cys7xb3wgihe36te3f3dmu7n7jfmsm3ot7lvunw53h4wzyozmy4y.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp22 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (24*r2) + (150528*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr4 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr5 + (x0 + (24*r2) + (150528*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp14, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr3 + (x3), tmp19, xmask)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr4 + (x3), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4z/c4zwnjnqam7lhc6nu7mhv56ogujtfvo3hb4xlixja5k5i52pnkde.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_82', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/a4/ca4gup6tsihp73niifzaxd2txmvq4bhl6zrnhaq3jgffgldjsdaq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_83', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/eo/ceoariuchyuomikogynswqlsnazn3bscsy4myzolzxv5milosb3g.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_84', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/5j/c5jjn5zlt4utsh3d6an4zm6wdaylleevgvuohq4tvfqbuxq5wtah.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7056
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
        tmp0 = tl.load(in_ptr0 + (x0 + (72*r2) + (9216*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (225792*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (72*r2) + (9216*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/i4/ci4uvsssm7el34vl3iilkp27xomjqy2ngml3vozyen2heyatyiwj.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_86', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/he/cheegdbnb6co7jfl6wmbrnq5nhwcago2mnxj4nep7cab6fykkkq6.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_87', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 24
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


# kernel path: /tmp/torchinductor_youkaichao/3p/c3p3cgrcsassjelivlkminrjz5vuxh6s4z3kdk75acgtl5vhkmmr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_88', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 98
    x1 = (xindex // 98)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (48*r2) + (6144*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x1) + (150528*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ry/cryt6j6d3jkdvnr5rd2p42j3cmety33nj2fhdf7nyyhrvcpuvi4b.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_89 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 48
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


# kernel path: /tmp/torchinductor_youkaichao/72/c72ruvl5ibyqo6lwieq3ladsnrabdrtvxezvbli5g2zxedhskv57.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 48
    x1 = (xindex // 48)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (48*r2) + (6144*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (150528*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (48*r2) + (6144*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7f/c7fpvfbon6qgyljbd3ifaoypbtn5ov77lskthgyojsxsmtvfc6i6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_91 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_91', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 48
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
        tmp0 = tl.load(in_ptr0 + (x0 + (48*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/sr/csrus7s3osmji3xjpphfhycplhyrkdh5iw5hzwsevpcifyrtk53q.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_92 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 48
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
    tmp0 = tl.load(in_ptr0 + (x2 + (48*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (48*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6m/c6mu2as3ojyukovd56p5cynswjciayoj653lq7um3jbxj5fdg4fy.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_93', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18816
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 392
    x1 = (xindex // 392)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (48*r2) + (6144*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x1) + (602112*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l2/cl2i7tsaa6qram3geytz7twuahunqvlllfmmtzyllrsn3fadaa3k.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_94 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_94', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 48
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


# kernel path: /tmp/torchinductor_youkaichao/pt/cptwo23vau4h3sc7o54iwx7jkk3xzcx7eunt2xvb6oeeg73gaztn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18816
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 48
    x1 = (xindex // 48)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (48*r2) + (6144*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x0) + (602112*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (48*r2) + (6144*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/c4/cc4kcn2uthguzzqhpiik2rnxikhip7hal4kggmyryp5k2ny7ncwx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_96 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_96', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 48
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
        tmp0 = tl.load(in_ptr0 + (x0 + (48*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/rs/crs3lbnu6th5ffu674ctxmtxzdsqkzpi5ccoukmxlu5ar2r375nf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_97 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_97', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 50176
    xnumel = 48
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
    tmp0 = tl.load(in_ptr0 + (x2 + (48*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (12544*x2) + (602112*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (48*y3)), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4n/c4nrcxnmrnkwuvlxpcbpt3jbig56hdflrtjtn2w2vsfvnvjknjpw.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_98', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/pd/cpdftes737xjaqjeq3a5aqg7boifodn2oqito6tfklpmlmuo4opi.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_99', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/lz/clzqwgdziwatar6fwleg7uyxlvi4zqo3xylk523v2n4kxqskcff4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_100 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_100', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ec/cec3rocc37xyqgn2qawke2pptpyvzzhyhibwz74i6ssrtzskm2hp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_101 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_101', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/5v/c5vg4yeieud2nigjidu46lm4h2oh7eil5iccn7dczpoxd3rfyv26.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_102 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_102', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/7p/c7pfu532ljuaysoe3nf6cdqwxq46fxjcbb6f45h72rxv5fzp555y.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_103 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_103', 'mutated_arg_names': []}
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
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (32*r2) + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x1) + (401408*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/if/cifxy5eiu2a26ccxk3zdzicizbnwxdis7664ndy3ez7hdnvkchha.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_104 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_104', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/or/corhpsl3n5ngvc3e7673o22p3uq72bcuunip6ebfefnh3gmbyzg7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_105 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_105', 'mutated_arg_names': []}
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
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x0) + (401408*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qc/cqcmeukt6prs5546mj4r2vmvc36k2it7cxknevkczxbtcsbs27ab.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_106 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_106', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/km/ckmzyzge2t5amqan56vla2zdvm45zrynh2fcr3ehf4yhyk6o52ax.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_107 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_107', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (12544*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (32*y3)), tmp11, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_159, primals_160, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, convolution, relu, convolution_1, relu_1, convolution_2, add_5, convolution_3, relu_2, convolution_4, relu_3, convolution_5, add_11, convolution_6, relu_4, convolution_7, relu_5, convolution_8, add_18, convolution_9, relu_6, convolution_10, relu_7, convolution_11, add_25, convolution_12, relu_8, convolution_13, relu_9, convolution_14, add_31, convolution_15, relu_10, convolution_16, relu_11, convolution_17, add_38, convolution_18, relu_12, convolution_19, relu_13, convolution_20, add_45, convolution_21, relu_14, convolution_22, relu_15, convolution_23, add_51, convolution_24, relu_16, convolution_25, relu_17, convolution_26, add_58, convolution_27, relu_18, convolution_28, relu_19, convolution_29, add_65, convolution_30, relu_20, convolution_31, relu_21, convolution_32, add_71, convolution_33, relu_22, convolution_34, relu_23, convolution_35, add_78, convolution_36, relu_24, convolution_37, relu_25, convolution_38, add_84, convolution_39, relu_26, convolution_40, relu_27, convolution_41, add_91, convolution_42, relu_28, convolution_43, relu_29, convolution_44, add_98, convolution_45, relu_30, convolution_46, relu_31, convolution_47, add_105, convolution_48, relu_32, convolution_49, relu_33, convolution_50, add_111, convolution_51, mean, permute_1, le, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_4, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_7, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_8, (16, ), (1, ))
    assert_size_stride(primals_10, (48, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_11, (48, ), (1, ))
    assert_size_stride(primals_13, (48, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_14, (48, ), (1, ))
    assert_size_stride(primals_16, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_19, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_20, (72, ), (1, ))
    assert_size_stride(primals_22, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (72, ), (1, ))
    assert_size_stride(primals_25, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_26, (24, ), (1, ))
    assert_size_stride(primals_28, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_29, (72, ), (1, ))
    assert_size_stride(primals_31, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_32, (72, ), (1, ))
    assert_size_stride(primals_34, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_35, (24, ), (1, ))
    assert_size_stride(primals_37, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_38, (72, ), (1, ))
    assert_size_stride(primals_40, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_41, (72, ), (1, ))
    assert_size_stride(primals_43, (40, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_44, (40, ), (1, ))
    assert_size_stride(primals_46, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_47, (120, ), (1, ))
    assert_size_stride(primals_49, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_50, (120, ), (1, ))
    assert_size_stride(primals_52, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_53, (40, ), (1, ))
    assert_size_stride(primals_55, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_56, (120, ), (1, ))
    assert_size_stride(primals_58, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_59, (120, ), (1, ))
    assert_size_stride(primals_61, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_62, (40, ), (1, ))
    assert_size_stride(primals_64, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_65, (240, ), (1, ))
    assert_size_stride(primals_67, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_68, (240, ), (1, ))
    assert_size_stride(primals_70, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_71, (80, ), (1, ))
    assert_size_stride(primals_73, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_74, (480, ), (1, ))
    assert_size_stride(primals_76, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_77, (480, ), (1, ))
    assert_size_stride(primals_79, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_80, (80, ), (1, ))
    assert_size_stride(primals_82, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_83, (480, ), (1, ))
    assert_size_stride(primals_85, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_86, (480, ), (1, ))
    assert_size_stride(primals_88, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_89, (80, ), (1, ))
    assert_size_stride(primals_91, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_92, (480, ), (1, ))
    assert_size_stride(primals_94, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_95, (480, ), (1, ))
    assert_size_stride(primals_97, (96, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_98, (96, ), (1, ))
    assert_size_stride(primals_100, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_101, (576, ), (1, ))
    assert_size_stride(primals_103, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_104, (576, ), (1, ))
    assert_size_stride(primals_106, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_107, (96, ), (1, ))
    assert_size_stride(primals_109, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_110, (576, ), (1, ))
    assert_size_stride(primals_112, (576, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_113, (576, ), (1, ))
    assert_size_stride(primals_115, (192, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_116, (192, ), (1, ))
    assert_size_stride(primals_118, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_119, (1152, ), (1, ))
    assert_size_stride(primals_121, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_122, (1152, ), (1, ))
    assert_size_stride(primals_124, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_125, (192, ), (1, ))
    assert_size_stride(primals_127, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_128, (1152, ), (1, ))
    assert_size_stride(primals_130, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_131, (1152, ), (1, ))
    assert_size_stride(primals_133, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_134, (192, ), (1, ))
    assert_size_stride(primals_136, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_137, (1152, ), (1, ))
    assert_size_stride(primals_139, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_140, (1152, ), (1, ))
    assert_size_stride(primals_142, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_143, (192, ), (1, ))
    assert_size_stride(primals_145, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_146, (1152, ), (1, ))
    assert_size_stride(primals_148, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_149, (1152, ), (1, ))
    assert_size_stride(primals_151, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_152, (320, ), (1, ))
    assert_size_stride(primals_154, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_155, (1280, ), (1, ))
    assert_size_stride(primals_159, (32, ), (1, ))
    assert_size_stride(primals_160, (32, ), (1, ))
    assert_size_stride(primals_162, (32, ), (1, ))
    assert_size_stride(primals_163, (32, ), (1, ))
    assert_size_stride(primals_165, (16, ), (1, ))
    assert_size_stride(primals_166, (16, ), (1, ))
    assert_size_stride(primals_168, (48, ), (1, ))
    assert_size_stride(primals_169, (48, ), (1, ))
    assert_size_stride(primals_171, (48, ), (1, ))
    assert_size_stride(primals_172, (48, ), (1, ))
    assert_size_stride(primals_174, (24, ), (1, ))
    assert_size_stride(primals_175, (24, ), (1, ))
    assert_size_stride(primals_177, (72, ), (1, ))
    assert_size_stride(primals_178, (72, ), (1, ))
    assert_size_stride(primals_180, (72, ), (1, ))
    assert_size_stride(primals_181, (72, ), (1, ))
    assert_size_stride(primals_183, (24, ), (1, ))
    assert_size_stride(primals_184, (24, ), (1, ))
    assert_size_stride(primals_186, (72, ), (1, ))
    assert_size_stride(primals_187, (72, ), (1, ))
    assert_size_stride(primals_189, (72, ), (1, ))
    assert_size_stride(primals_190, (72, ), (1, ))
    assert_size_stride(primals_192, (24, ), (1, ))
    assert_size_stride(primals_193, (24, ), (1, ))
    assert_size_stride(primals_195, (72, ), (1, ))
    assert_size_stride(primals_196, (72, ), (1, ))
    assert_size_stride(primals_198, (72, ), (1, ))
    assert_size_stride(primals_199, (72, ), (1, ))
    assert_size_stride(primals_201, (40, ), (1, ))
    assert_size_stride(primals_202, (40, ), (1, ))
    assert_size_stride(primals_204, (120, ), (1, ))
    assert_size_stride(primals_205, (120, ), (1, ))
    assert_size_stride(primals_207, (120, ), (1, ))
    assert_size_stride(primals_208, (120, ), (1, ))
    assert_size_stride(primals_210, (40, ), (1, ))
    assert_size_stride(primals_211, (40, ), (1, ))
    assert_size_stride(primals_213, (120, ), (1, ))
    assert_size_stride(primals_214, (120, ), (1, ))
    assert_size_stride(primals_216, (120, ), (1, ))
    assert_size_stride(primals_217, (120, ), (1, ))
    assert_size_stride(primals_219, (40, ), (1, ))
    assert_size_stride(primals_220, (40, ), (1, ))
    assert_size_stride(primals_222, (240, ), (1, ))
    assert_size_stride(primals_223, (240, ), (1, ))
    assert_size_stride(primals_225, (240, ), (1, ))
    assert_size_stride(primals_226, (240, ), (1, ))
    assert_size_stride(primals_228, (80, ), (1, ))
    assert_size_stride(primals_229, (80, ), (1, ))
    assert_size_stride(primals_231, (480, ), (1, ))
    assert_size_stride(primals_232, (480, ), (1, ))
    assert_size_stride(primals_234, (480, ), (1, ))
    assert_size_stride(primals_235, (480, ), (1, ))
    assert_size_stride(primals_237, (80, ), (1, ))
    assert_size_stride(primals_238, (80, ), (1, ))
    assert_size_stride(primals_240, (480, ), (1, ))
    assert_size_stride(primals_241, (480, ), (1, ))
    assert_size_stride(primals_243, (480, ), (1, ))
    assert_size_stride(primals_244, (480, ), (1, ))
    assert_size_stride(primals_246, (80, ), (1, ))
    assert_size_stride(primals_247, (80, ), (1, ))
    assert_size_stride(primals_249, (480, ), (1, ))
    assert_size_stride(primals_250, (480, ), (1, ))
    assert_size_stride(primals_252, (480, ), (1, ))
    assert_size_stride(primals_253, (480, ), (1, ))
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
    assert_size_stride(primals_273, (192, ), (1, ))
    assert_size_stride(primals_274, (192, ), (1, ))
    assert_size_stride(primals_276, (1152, ), (1, ))
    assert_size_stride(primals_277, (1152, ), (1, ))
    assert_size_stride(primals_279, (1152, ), (1, ))
    assert_size_stride(primals_280, (1152, ), (1, ))
    assert_size_stride(primals_282, (192, ), (1, ))
    assert_size_stride(primals_283, (192, ), (1, ))
    assert_size_stride(primals_285, (1152, ), (1, ))
    assert_size_stride(primals_286, (1152, ), (1, ))
    assert_size_stride(primals_288, (1152, ), (1, ))
    assert_size_stride(primals_289, (1152, ), (1, ))
    assert_size_stride(primals_291, (192, ), (1, ))
    assert_size_stride(primals_292, (192, ), (1, ))
    assert_size_stride(primals_294, (1152, ), (1, ))
    assert_size_stride(primals_295, (1152, ), (1, ))
    assert_size_stride(primals_297, (1152, ), (1, ))
    assert_size_stride(primals_298, (1152, ), (1, ))
    assert_size_stride(primals_300, (192, ), (1, ))
    assert_size_stride(primals_301, (192, ), (1, ))
    assert_size_stride(primals_303, (1152, ), (1, ))
    assert_size_stride(primals_304, (1152, ), (1, ))
    assert_size_stride(primals_306, (1152, ), (1, ))
    assert_size_stride(primals_307, (1152, ), (1, ))
    assert_size_stride(primals_309, (320, ), (1, ))
    assert_size_stride(primals_310, (320, ), (1, ))
    assert_size_stride(primals_312, (1280, ), (1, ))
    assert_size_stride(primals_313, (1280, ), (1, ))
    assert_size_stride(primals_315, (4, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(relu, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_1, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(relu_1, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_2, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(add_5, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_3, (4, 48, 112, 112), (602112, 1, 5376, 48))
    assert_size_stride(relu_2, (4, 48, 112, 112), (602112, 1, 5376, 48))
    assert_size_stride(convolution_4, (4, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(relu_3, (4, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(convolution_5, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(add_11, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_6, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(relu_4, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_7, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(relu_5, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_8, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(add_18, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_9, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(relu_6, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_10, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(relu_7, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_11, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(add_25, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_12, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(relu_8, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_13, (4, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(relu_9, (4, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(convolution_14, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(add_31, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_15, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(relu_10, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_16, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(relu_11, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_17, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(add_38, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_18, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(relu_12, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_19, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(relu_13, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_20, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(add_45, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_21, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(relu_14, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(convolution_22, (4, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(relu_15, (4, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(convolution_23, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(add_51, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_24, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(relu_16, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_25, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(relu_17, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_26, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(add_58, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_27, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(relu_18, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_28, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(relu_19, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_29, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(add_65, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_30, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(relu_20, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_31, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(relu_21, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_32, (4, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(add_71, (4, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(convolution_33, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(relu_22, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_34, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(relu_23, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_35, (4, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(add_78, (4, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(convolution_36, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(relu_24, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_37, (4, 576, 7, 7), (28224, 1, 4032, 576))
    assert_size_stride(relu_25, (4, 576, 7, 7), (28224, 1, 4032, 576))
    assert_size_stride(convolution_38, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(add_84, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_39, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(relu_26, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_40, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(relu_27, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_41, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(add_91, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_42, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(relu_28, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_43, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(relu_29, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_44, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(add_98, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_45, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(relu_30, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_46, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(relu_31, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_47, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(add_105, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_48, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(relu_32, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_49, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(relu_33, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_50, (4, 320, 7, 7), (15680, 1, 2240, 320))
    assert_size_stride(add_111, (4, 320, 7, 7), (15680, 1, 2240, 320))
    assert_size_stride(convolution_51, (4, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(mean, (4, 1280), (1280, 1))
    assert_size_stride(permute_1, (1000, 1280), (1280, 1))
    assert_size_stride(le, (4, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), mean, out=buf1)
        del mean
        buf2 = empty((1000, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_sum_view_0.run(tangents_1, buf2, 1000, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty_strided((1280, 2), (1, 1280), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1280, 2), (1, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_div_native_batch_norm_backward_threshold_backward_1.run(le, buf0, convolution_51, primals_312, buf3, buf5, 2560, 98, grid=grid(2560), stream=stream0)
        del convolution_51
        del primals_312
        buf4 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_2.run(buf3, buf4, 1280, 2, grid=grid(1280), stream=stream0)
        del buf3
        buf6 = empty((1280, ), device='cuda', dtype=torch.float32)
        buf7 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_3.run(buf7, buf5, primals_313, 1280, 2, grid=grid(1280), stream=stream0)
        del buf5
        buf8 = empty_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4.run(le, buf0, primals_313, primals_155, buf8, 250880, grid=grid(250880), stream=stream0)
        del buf0
        del le
        del primals_155
        del primals_313
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
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
        buf17 = aten.convolution_backward(buf16, relu_33, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_151
        buf18 = buf17[0]
        buf19 = buf17[1]
        del buf17
        buf20 = empty_strided((1152, 2), (1, 1152), device='cuda', dtype=torch.float32)
        buf22 = empty_strided((1152, 2), (1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_33, buf18, convolution_49, primals_306, buf20, buf22, 2304, 98, grid=grid(2304), stream=stream0)
        del convolution_49
        del primals_306
        buf21 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf20, buf21, 1152, 2, grid=grid(1152), stream=stream0)
        buf23 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf24 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf24, buf22, primals_307, 1152, 2, grid=grid(1152), stream=stream0)
        buf25 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_33, buf18, primals_307, primals_149, buf25, 196, 1152, grid=grid(196, 1152), stream=stream0)
        del buf18
        del primals_149
        del primals_307
        del relu_33
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf26 = aten.convolution_backward(buf25, relu_32, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1152, [True, True, False])
        del primals_148
        buf27 = buf26[0]
        buf28 = buf26[1]
        del buf26
        buf29 = buf22; del buf22  # reuse
        buf31 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_32, buf27, convolution_48, primals_303, buf29, buf31, 2304, 98, grid=grid(2304), stream=stream0)
        del convolution_48
        del primals_303
        buf30 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf29, buf30, 1152, 2, grid=grid(1152), stream=stream0)
        buf32 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf33 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf33, buf31, primals_304, 1152, 2, grid=grid(1152), stream=stream0)
        buf34 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_32, buf27, primals_304, primals_146, buf34, 196, 1152, grid=grid(196, 1152), stream=stream0)
        del buf27
        del primals_146
        del primals_304
        del relu_32
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf35 = aten.convolution_backward(buf34, add_105, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_105
        del primals_145
        buf36 = buf35[0]
        buf37 = buf35[1]
        del buf35
        buf38 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_13.run(buf36, buf38, 192, 196, grid=grid(192), stream=stream0)
        buf39 = empty_strided((192, 2), (1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_14.run(buf36, convolution_47, primals_300, buf39, 384, 98, grid=grid(384), stream=stream0)
        del convolution_47
        del primals_300
        buf40 = empty((192, ), device='cuda', dtype=torch.float32)
        buf41 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_15.run(buf41, buf39, primals_301, 192, 2, grid=grid(192), stream=stream0)
        del buf39
        buf42 = empty((4, 192, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_16.run(buf36, primals_301, primals_143, buf42, 37632, grid=grid(37632), stream=stream0)
        del primals_143
        del primals_301
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf43 = aten.convolution_backward(buf42, relu_31, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_142
        buf44 = buf43[0]
        buf45 = buf43[1]
        del buf43
        buf46 = buf31; del buf31  # reuse
        buf48 = buf29; del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_31, buf44, convolution_46, primals_297, buf46, buf48, 2304, 98, grid=grid(2304), stream=stream0)
        del convolution_46
        del primals_297
        buf47 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf46, buf47, 1152, 2, grid=grid(1152), stream=stream0)
        buf49 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf50 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf50, buf48, primals_298, 1152, 2, grid=grid(1152), stream=stream0)
        buf51 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_31, buf44, primals_298, primals_140, buf51, 196, 1152, grid=grid(196, 1152), stream=stream0)
        del buf44
        del primals_140
        del primals_298
        del relu_31
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf52 = aten.convolution_backward(buf51, relu_30, primals_139, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
        del primals_139
        buf53 = buf52[0]
        buf54 = buf52[1]
        del buf52
        buf55 = buf48; del buf48  # reuse
        buf57 = buf46; del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_30, buf53, convolution_45, primals_294, buf55, buf57, 2304, 98, grid=grid(2304), stream=stream0)
        del convolution_45
        del primals_294
        buf56 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf55, buf56, 1152, 2, grid=grid(1152), stream=stream0)
        buf58 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf59 = buf58; del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf59, buf57, primals_295, 1152, 2, grid=grid(1152), stream=stream0)
        buf60 = buf51; del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_30, buf53, primals_295, primals_137, buf60, 196, 1152, grid=grid(196, 1152), stream=stream0)
        del buf53
        del primals_137
        del primals_295
        del relu_30
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf61 = aten.convolution_backward(buf60, add_98, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_98
        del primals_136
        buf62 = buf61[0]
        buf63 = buf61[1]
        del buf61
        buf67 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_17.run(buf36, buf62, primals_292, primals_134, buf67, 37632, grid=grid(37632), stream=stream0)
        del primals_134
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf68 = aten.convolution_backward(buf67, relu_29, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_133
        buf69 = buf68[0]
        buf76 = buf60; del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_29, buf69, primals_289, primals_131, buf76, 196, 1152, grid=grid(196, 1152), stream=stream0)
        del primals_131
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf77 = aten.convolution_backward(buf76, relu_28, primals_130, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
        del primals_130
        buf78 = buf77[0]
        buf85 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_28, buf78, primals_286, primals_128, buf85, 196, 1152, grid=grid(196, 1152), stream=stream0)
        del primals_128
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf86 = aten.convolution_backward(buf85, add_91, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_91
        del primals_127
        buf87 = buf86[0]
        buf92 = buf67; del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_18.run(buf36, buf62, buf87, primals_283, primals_125, buf92, 37632, grid=grid(37632), stream=stream0)
        del primals_125
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf93 = aten.convolution_backward(buf92, relu_27, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf92
        del primals_124
        buf94 = buf93[0]
        buf101 = buf85; del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_27, buf94, primals_280, primals_122, buf101, 196, 1152, grid=grid(196, 1152), stream=stream0)
        del primals_122
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf102 = aten.convolution_backward(buf101, relu_26, primals_121, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
        del primals_121
        buf103 = buf102[0]
        buf110 = buf101; del buf101  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_26, buf103, primals_277, primals_119, buf110, 196, 1152, grid=grid(196, 1152), stream=stream0)
        del primals_119
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf111 = aten.convolution_backward(buf110, add_84, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_84
        del buf110
        del primals_118
        buf112 = buf111[0]
        buf64 = empty((192, ), device='cuda', dtype=torch.float32)
        buf65 = empty((192, ), device='cuda', dtype=torch.float32)
        buf89 = empty((192, ), device='cuda', dtype=torch.float32)
        buf90 = empty((192, ), device='cuda', dtype=torch.float32)
        buf114 = empty((192, ), device='cuda', dtype=torch.float32)
        buf115 = empty((192, ), device='cuda', dtype=torch.float32)
        buf66 = buf65; del buf65  # reuse
        buf91 = buf90; del buf90  # reuse
        buf116 = buf115; del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_19.run(buf66, buf91, buf116, buf36, buf62, convolution_44, primals_291, buf87, convolution_41, primals_282, buf112, convolution_38, primals_273, primals_292, primals_283, primals_274, buf64, buf89, buf114, 192, 196, grid=grid(192), stream=stream0)
        del convolution_38
        del convolution_41
        del convolution_44
        del primals_273
        del primals_282
        del primals_283
        del primals_291
        del primals_292
        buf70 = buf68[1]
        del buf68
        buf71 = buf57; del buf57  # reuse
        buf73 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_29, buf69, convolution_43, primals_288, buf71, buf73, 2304, 98, grid=grid(2304), stream=stream0)
        del buf69
        del convolution_43
        del primals_288
        del relu_29
        buf72 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf71, buf72, 1152, 2, grid=grid(1152), stream=stream0)
        buf74 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf75 = buf74; del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf75, buf73, primals_289, 1152, 2, grid=grid(1152), stream=stream0)
        del primals_289
        buf79 = buf77[1]
        del buf77
        buf80 = buf73; del buf73  # reuse
        buf82 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_28, buf78, convolution_42, primals_285, buf80, buf82, 2304, 98, grid=grid(2304), stream=stream0)
        del buf78
        del convolution_42
        del primals_285
        del relu_28
        buf81 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf80, buf81, 1152, 2, grid=grid(1152), stream=stream0)
        buf83 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf84 = buf83; del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf84, buf82, primals_286, 1152, 2, grid=grid(1152), stream=stream0)
        del primals_286
        buf88 = buf86[1]
        del buf86
        buf95 = buf93[1]
        del buf93
        buf96 = buf82; del buf82  # reuse
        buf98 = buf80; del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_27, buf94, convolution_40, primals_279, buf96, buf98, 2304, 98, grid=grid(2304), stream=stream0)
        del buf94
        del convolution_40
        del primals_279
        del relu_27
        buf97 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf96, buf97, 1152, 2, grid=grid(1152), stream=stream0)
        buf99 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf100 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf100, buf98, primals_280, 1152, 2, grid=grid(1152), stream=stream0)
        del primals_280
        buf104 = buf102[1]
        del buf102
        buf105 = buf98; del buf98  # reuse
        buf107 = buf96; del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_26, buf103, convolution_39, primals_276, buf105, buf107, 2304, 98, grid=grid(2304), stream=stream0)
        del convolution_39
        del primals_276
        del relu_26
        buf106 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf105, buf106, 1152, 2, grid=grid(1152), stream=stream0)
        del buf105
        buf108 = empty((1152, ), device='cuda', dtype=torch.float32)
        buf109 = buf108; del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf109, buf107, primals_277, 1152, 2, grid=grid(1152), stream=stream0)
        del buf107
        del primals_277
        buf113 = buf111[1]
        del buf111
        buf117 = buf112; del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_20.run(buf117, buf36, buf62, buf87, primals_274, primals_116, 37632, grid=grid(37632), stream=stream0)
        del buf36
        del buf62
        del buf87
        del primals_116
        del primals_274
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf118 = aten.convolution_backward(buf117, relu_25, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf117
        del primals_115
        buf119 = buf118[0]
        buf120 = buf118[1]
        del buf118
        buf121 = empty_strided((576, 2), (1, 576), device='cuda', dtype=torch.float32)
        buf123 = empty_strided((576, 2), (1, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_21.run(relu_25, buf119, convolution_37, primals_270, buf121, buf123, 1152, 98, grid=grid(1152), stream=stream0)
        del convolution_37
        del primals_270
        buf122 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_22.run(buf121, buf122, 576, 2, grid=grid(576), stream=stream0)
        del buf121
        buf124 = empty((576, ), device='cuda', dtype=torch.float32)
        buf125 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_23.run(buf125, buf123, primals_271, 576, 2, grid=grid(576), stream=stream0)
        del buf123
        buf126 = empty_strided((4, 576, 7, 7), (28224, 1, 4032, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_24.run(relu_25, buf119, primals_271, primals_113, buf126, 196, 576, grid=grid(196, 576), stream=stream0)
        del buf119
        del primals_113
        del primals_271
        del relu_25
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf127 = aten.convolution_backward(buf126, relu_24, primals_112, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 576, [True, True, False])
        del buf126
        del primals_112
        buf128 = buf127[0]
        buf129 = buf127[1]
        del buf127
        buf130 = empty((576, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(relu_24, buf128, buf130, 4032, 112, grid=grid(4032), stream=stream0)
        buf131 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_26.run(buf130, buf131, 576, 7, grid=grid(576), stream=stream0)
        buf132 = reinterpret_tensor(buf130, (576, 7), (1, 576), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_24, buf128, convolution_36, primals_267, buf132, 4032, 112, grid=grid(4032), stream=stream0)
        del convolution_36
        del primals_267
        buf133 = empty((576, ), device='cuda', dtype=torch.float32)
        buf134 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf134, buf132, primals_268, 576, 7, grid=grid(576), stream=stream0)
        buf135 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(relu_24, buf128, primals_268, primals_110, buf135, 784, 576, grid=grid(784, 576), stream=stream0)
        del buf128
        del primals_110
        del primals_268
        del relu_24
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf136 = aten.convolution_backward(buf135, add_78, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_78
        del primals_109
        buf137 = buf136[0]
        buf138 = buf136[1]
        del buf136
        buf143 = empty((4, 96, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_30.run(buf137, primals_265, primals_107, buf143, 75264, grid=grid(75264), stream=stream0)
        del primals_107
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf144 = aten.convolution_backward(buf143, relu_23, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf143
        del primals_106
        buf145 = buf144[0]
        buf152 = buf135; del buf135  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(relu_23, buf145, primals_262, primals_104, buf152, 784, 576, grid=grid(784, 576), stream=stream0)
        del primals_104
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf153 = aten.convolution_backward(buf152, relu_22, primals_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False])
        del primals_103
        buf154 = buf153[0]
        buf161 = buf152; del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(relu_22, buf154, primals_259, primals_101, buf161, 784, 576, grid=grid(784, 576), stream=stream0)
        del primals_101
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf162 = aten.convolution_backward(buf161, add_71, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_71
        del buf161
        del primals_100
        buf163 = buf162[0]
        buf139 = empty((96, ), device='cuda', dtype=torch.float32)
        buf165 = empty((96, ), device='cuda', dtype=torch.float32)
        buf166 = empty((96, ), device='cuda', dtype=torch.float32)
        buf167 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_31.run(buf167, buf137, buf163, convolution_32, primals_255, primals_256, buf139, buf165, 96, 784, grid=grid(96), stream=stream0)
        del convolution_32
        del primals_255
        buf140 = empty((96, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf137, convolution_35, primals_264, buf140, 672, 112, grid=grid(672), stream=stream0)
        del convolution_35
        del primals_264
        buf141 = empty((96, ), device='cuda', dtype=torch.float32)
        buf142 = buf141; del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_33.run(buf142, buf140, primals_265, 96, 7, grid=grid(96), stream=stream0)
        del buf140
        del primals_265
        buf146 = buf144[1]
        del buf144
        buf147 = reinterpret_tensor(buf132, (576, 7), (7, 1), 0); del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(relu_23, buf145, buf147, 4032, 112, grid=grid(4032), stream=stream0)
        buf148 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_26.run(buf147, buf148, 576, 7, grid=grid(576), stream=stream0)
        buf149 = reinterpret_tensor(buf147, (576, 7), (1, 576), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_34.run(relu_23, buf145, convolution_34, primals_261, buf149, 4032, 112, grid=grid(4032), stream=stream0)
        del buf145
        del convolution_34
        del primals_261
        del relu_23
        buf150 = empty((576, ), device='cuda', dtype=torch.float32)
        buf151 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf151, buf149, primals_262, 576, 7, grid=grid(576), stream=stream0)
        del primals_262
        buf155 = buf153[1]
        del buf153
        buf156 = reinterpret_tensor(buf149, (576, 7), (7, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(relu_22, buf154, buf156, 4032, 112, grid=grid(4032), stream=stream0)
        buf157 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_26.run(buf156, buf157, 576, 7, grid=grid(576), stream=stream0)
        buf158 = reinterpret_tensor(buf156, (576, 7), (1, 576), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_34.run(relu_22, buf154, convolution_33, primals_258, buf158, 4032, 112, grid=grid(4032), stream=stream0)
        del buf154
        del convolution_33
        del primals_258
        del relu_22
        buf159 = empty((576, ), device='cuda', dtype=torch.float32)
        buf160 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_28.run(buf160, buf158, primals_259, 576, 7, grid=grid(576), stream=stream0)
        del buf158
        del primals_259
        buf164 = buf162[1]
        del buf162
        buf168 = buf137; del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_35.run(buf168, buf163, primals_256, primals_98, 75264, grid=grid(75264), stream=stream0)
        del buf163
        del primals_256
        del primals_98
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf169 = aten.convolution_backward(buf168, relu_21, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf168
        del primals_97
        buf170 = buf169[0]
        buf171 = buf169[1]
        del buf169
        buf172 = empty((480, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_21, buf170, buf172, 3360, 112, grid=grid(3360), stream=stream0)
        buf173 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf172, buf173, 480, 7, grid=grid(480), stream=stream0)
        buf174 = reinterpret_tensor(buf172, (480, 7), (1, 480), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_38.run(relu_21, buf170, convolution_31, primals_252, buf174, 3360, 112, grid=grid(3360), stream=stream0)
        del convolution_31
        del primals_252
        buf175 = empty((480, ), device='cuda', dtype=torch.float32)
        buf176 = buf175; del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_39.run(buf176, buf174, primals_253, 480, 7, grid=grid(480), stream=stream0)
        buf177 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40.run(relu_21, buf170, primals_253, primals_95, buf177, 784, 480, grid=grid(784, 480), stream=stream0)
        del buf170
        del primals_253
        del primals_95
        del relu_21
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf178 = aten.convolution_backward(buf177, relu_20, primals_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
        del primals_94
        buf179 = buf178[0]
        buf180 = buf178[1]
        del buf178
        buf181 = reinterpret_tensor(buf174, (480, 7), (7, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_20, buf179, buf181, 3360, 112, grid=grid(3360), stream=stream0)
        buf182 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf181, buf182, 480, 7, grid=grid(480), stream=stream0)
        buf183 = reinterpret_tensor(buf181, (480, 7), (1, 480), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_38.run(relu_20, buf179, convolution_30, primals_249, buf183, 3360, 112, grid=grid(3360), stream=stream0)
        del convolution_30
        del primals_249
        buf184 = empty((480, ), device='cuda', dtype=torch.float32)
        buf185 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_39.run(buf185, buf183, primals_250, 480, 7, grid=grid(480), stream=stream0)
        buf186 = buf177; del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40.run(relu_20, buf179, primals_250, primals_92, buf186, 784, 480, grid=grid(784, 480), stream=stream0)
        del buf179
        del primals_250
        del primals_92
        del relu_20
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf187 = aten.convolution_backward(buf186, add_65, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_65
        del primals_91
        buf188 = buf187[0]
        buf189 = buf187[1]
        del buf187
        buf194 = reinterpret_tensor(buf16, (4, 80, 14, 14), (15680, 196, 14, 1), 0); del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_41.run(buf188, primals_247, primals_89, buf194, 62720, grid=grid(62720), stream=stream0)
        del primals_89
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf195 = aten.convolution_backward(buf194, relu_19, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_88
        buf196 = buf195[0]
        buf203 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40.run(relu_19, buf196, primals_244, primals_86, buf203, 784, 480, grid=grid(784, 480), stream=stream0)
        del primals_86
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf204 = aten.convolution_backward(buf203, relu_18, primals_85, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False])
        del primals_85
        buf205 = buf204[0]
        buf212 = buf203; del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40.run(relu_18, buf205, primals_241, primals_83, buf212, 784, 480, grid=grid(784, 480), stream=stream0)
        del primals_83
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf213 = aten.convolution_backward(buf212, add_58, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_58
        del primals_82
        buf214 = buf213[0]
        buf219 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_42.run(buf188, buf214, primals_238, primals_80, buf219, 62720, grid=grid(62720), stream=stream0)
        del primals_80
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf220 = aten.convolution_backward(buf219, relu_17, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf219
        del primals_79
        buf221 = buf220[0]
        buf228 = buf212; del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40.run(relu_17, buf221, primals_235, primals_77, buf228, 784, 480, grid=grid(784, 480), stream=stream0)
        del primals_77
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf229 = aten.convolution_backward(buf228, relu_16, primals_76, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False])
        del primals_76
        buf230 = buf229[0]
        buf237 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40.run(relu_16, buf230, primals_232, primals_74, buf237, 784, 480, grid=grid(784, 480), stream=stream0)
        del primals_74
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf238 = aten.convolution_backward(buf237, add_51, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_51
        del buf237
        del primals_73
        buf239 = buf238[0]
        buf190 = empty((80, ), device='cuda', dtype=torch.float32)
        buf216 = empty((80, ), device='cuda', dtype=torch.float32)
        buf217 = empty((80, ), device='cuda', dtype=torch.float32)
        buf241 = empty((80, ), device='cuda', dtype=torch.float32)
        buf242 = empty((80, ), device='cuda', dtype=torch.float32)
        buf218 = buf217; del buf217  # reuse
        buf243 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_43.run(buf218, buf243, buf188, buf214, convolution_26, primals_237, buf239, convolution_23, primals_228, primals_238, primals_229, buf190, buf216, buf241, 80, 784, grid=grid(80), stream=stream0)
        del convolution_23
        del convolution_26
        del primals_228
        del primals_237
        del primals_238
        buf191 = empty((80, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_44.run(buf188, convolution_29, primals_246, buf191, 560, 112, grid=grid(560), stream=stream0)
        del convolution_29
        del primals_246
        buf192 = empty((80, ), device='cuda', dtype=torch.float32)
        buf193 = buf192; del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_45.run(buf193, buf191, primals_247, 80, 7, grid=grid(80), stream=stream0)
        del buf191
        del primals_247
        buf197 = buf195[1]
        del buf195
        buf198 = reinterpret_tensor(buf183, (480, 7), (7, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_19, buf196, buf198, 3360, 112, grid=grid(3360), stream=stream0)
        buf199 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf198, buf199, 480, 7, grid=grid(480), stream=stream0)
        buf200 = reinterpret_tensor(buf198, (480, 7), (1, 480), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_46.run(relu_19, buf196, convolution_28, primals_243, buf200, 3360, 112, grid=grid(3360), stream=stream0)
        del buf196
        del convolution_28
        del primals_243
        del relu_19
        buf201 = empty((480, ), device='cuda', dtype=torch.float32)
        buf202 = buf201; del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_39.run(buf202, buf200, primals_244, 480, 7, grid=grid(480), stream=stream0)
        del primals_244
        buf206 = buf204[1]
        del buf204
        buf207 = reinterpret_tensor(buf200, (480, 7), (7, 1), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_18, buf205, buf207, 3360, 112, grid=grid(3360), stream=stream0)
        buf208 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf207, buf208, 480, 7, grid=grid(480), stream=stream0)
        buf209 = reinterpret_tensor(buf207, (480, 7), (1, 480), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_46.run(relu_18, buf205, convolution_27, primals_240, buf209, 3360, 112, grid=grid(3360), stream=stream0)
        del buf205
        del convolution_27
        del primals_240
        del relu_18
        buf210 = empty((480, ), device='cuda', dtype=torch.float32)
        buf211 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_39.run(buf211, buf209, primals_241, 480, 7, grid=grid(480), stream=stream0)
        del primals_241
        buf215 = buf213[1]
        del buf213
        buf222 = buf220[1]
        del buf220
        buf223 = reinterpret_tensor(buf209, (480, 7), (7, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_17, buf221, buf223, 3360, 112, grid=grid(3360), stream=stream0)
        buf224 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf223, buf224, 480, 7, grid=grid(480), stream=stream0)
        buf225 = reinterpret_tensor(buf223, (480, 7), (1, 480), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_46.run(relu_17, buf221, convolution_25, primals_234, buf225, 3360, 112, grid=grid(3360), stream=stream0)
        del buf221
        del convolution_25
        del primals_234
        del relu_17
        buf226 = empty((480, ), device='cuda', dtype=torch.float32)
        buf227 = buf226; del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_39.run(buf227, buf225, primals_235, 480, 7, grid=grid(480), stream=stream0)
        del primals_235
        buf231 = buf229[1]
        del buf229
        buf232 = reinterpret_tensor(buf225, (480, 7), (7, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_16, buf230, buf232, 3360, 112, grid=grid(3360), stream=stream0)
        buf233 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf232, buf233, 480, 7, grid=grid(480), stream=stream0)
        buf234 = reinterpret_tensor(buf232, (480, 7), (1, 480), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_46.run(relu_16, buf230, convolution_24, primals_231, buf234, 3360, 112, grid=grid(3360), stream=stream0)
        del convolution_24
        del primals_231
        del relu_16
        buf235 = empty((480, ), device='cuda', dtype=torch.float32)
        buf236 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_39.run(buf236, buf234, primals_232, 480, 7, grid=grid(480), stream=stream0)
        del buf234
        del primals_232
        buf240 = buf238[1]
        del buf238
        buf244 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_47.run(buf244, buf214, buf239, primals_229, primals_71, 62720, grid=grid(62720), stream=stream0)
        del buf214
        del buf239
        del primals_229
        del primals_71
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf245 = aten.convolution_backward(buf244, relu_15, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf244
        del primals_70
        buf246 = buf245[0]
        buf247 = buf245[1]
        del buf245
        buf248 = empty((240, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_48.run(relu_15, buf246, buf248, 1680, 112, grid=grid(1680), stream=stream0)
        buf249 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_49.run(buf248, buf249, 240, 7, grid=grid(240), stream=stream0)
        buf250 = reinterpret_tensor(buf248, (240, 7), (1, 240), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_50.run(relu_15, buf246, convolution_22, primals_225, buf250, 1680, 112, grid=grid(1680), stream=stream0)
        del convolution_22
        del primals_225
        buf251 = empty((240, ), device='cuda', dtype=torch.float32)
        buf252 = buf251; del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_51.run(buf252, buf250, primals_226, 240, 7, grid=grid(240), stream=stream0)
        del buf250
        buf253 = empty_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52.run(relu_15, buf246, primals_226, primals_68, buf253, 784, 240, grid=grid(784, 240), stream=stream0)
        del buf246
        del primals_226
        del primals_68
        del relu_15
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf254 = aten.convolution_backward(buf253, relu_14, primals_67, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False])
        del buf253
        del primals_67
        buf255 = buf254[0]
        buf256 = buf254[1]
        del buf254
        buf257 = empty((240, 25), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_53.run(relu_14, buf255, buf257, 6000, 126, grid=grid(6000), stream=stream0)
        buf258 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_54.run(buf257, buf258, 240, 25, grid=grid(240), stream=stream0)
        buf259 = reinterpret_tensor(buf257, (240, 25), (1, 240), 0); del buf257  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_55.run(relu_14, buf255, convolution_21, primals_222, buf259, 6000, 126, grid=grid(6000), stream=stream0)
        del convolution_21
        del primals_222
        buf260 = empty((240, ), device='cuda', dtype=torch.float32)
        buf261 = buf260; del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_56.run(buf261, buf259, primals_223, 240, 25, grid=grid(240), stream=stream0)
        del buf259
        buf262 = empty_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_57.run(relu_14, buf255, primals_223, primals_65, buf262, 3136, 240, grid=grid(3136, 240), stream=stream0)
        del buf255
        del primals_223
        del primals_65
        del relu_14
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf263 = aten.convolution_backward(buf262, add_45, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_45
        del buf262
        del primals_64
        buf264 = buf263[0]
        buf265 = buf263[1]
        del buf263
        buf270 = empty((4, 40, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_58.run(buf264, primals_220, primals_62, buf270, 125440, grid=grid(125440), stream=stream0)
        del primals_62
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf271 = aten.convolution_backward(buf270, relu_13, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_61
        buf272 = buf271[0]
        buf279 = reinterpret_tensor(buf230, (4, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_59.run(relu_13, buf272, primals_217, primals_59, buf279, 3136, 120, grid=grid(3136, 120), stream=stream0)
        del primals_59
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf280 = aten.convolution_backward(buf279, relu_12, primals_58, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del primals_58
        buf281 = buf280[0]
        buf288 = buf279; del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_59.run(relu_12, buf281, primals_214, primals_56, buf288, 3136, 120, grid=grid(3136, 120), stream=stream0)
        del primals_56
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf289 = aten.convolution_backward(buf288, add_38, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_38
        del primals_55
        buf290 = buf289[0]
        buf295 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_60.run(buf264, buf290, primals_211, primals_53, buf295, 125440, grid=grid(125440), stream=stream0)
        del primals_53
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf296 = aten.convolution_backward(buf295, relu_11, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf295
        del primals_52
        buf297 = buf296[0]
        buf304 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_59.run(relu_11, buf297, primals_208, primals_50, buf304, 3136, 120, grid=grid(3136, 120), stream=stream0)
        del primals_50
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf305 = aten.convolution_backward(buf304, relu_10, primals_49, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del primals_49
        buf306 = buf305[0]
        buf313 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_59.run(relu_10, buf306, primals_205, primals_47, buf313, 3136, 120, grid=grid(3136, 120), stream=stream0)
        del primals_47
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf314 = aten.convolution_backward(buf313, add_31, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_31
        del buf313
        del primals_46
        buf315 = buf314[0]
        buf266 = empty((40, ), device='cuda', dtype=torch.float32)
        buf292 = empty((40, ), device='cuda', dtype=torch.float32)
        buf293 = empty((40, ), device='cuda', dtype=torch.float32)
        buf317 = empty((40, ), device='cuda', dtype=torch.float32)
        buf318 = empty((40, ), device='cuda', dtype=torch.float32)
        buf294 = buf293; del buf293  # reuse
        buf319 = buf318; del buf318  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_61.run(buf294, buf319, buf264, buf290, convolution_17, primals_210, buf315, convolution_14, primals_201, primals_211, primals_202, buf266, buf292, buf317, 40, 3136, grid=grid(40), stream=stream0)
        del convolution_14
        del convolution_17
        del primals_201
        del primals_210
        del primals_211
        buf267 = empty((40, 25), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_62.run(buf264, convolution_20, primals_219, buf267, 1000, 126, grid=grid(1000), stream=stream0)
        del convolution_20
        del primals_219
        buf268 = empty((40, ), device='cuda', dtype=torch.float32)
        buf269 = buf268; del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_63.run(buf269, buf267, primals_220, 40, 25, grid=grid(40), stream=stream0)
        del buf267
        del primals_220
        buf273 = buf271[1]
        del buf271
        buf274 = empty((120, 25), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_64.run(relu_13, buf272, buf274, 3000, 126, grid=grid(3000), stream=stream0)
        buf275 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_65.run(buf274, buf275, 120, 25, grid=grid(120), stream=stream0)
        buf276 = reinterpret_tensor(buf274, (120, 25), (1, 120), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(relu_13, buf272, convolution_19, primals_216, buf276, 3000, 126, grid=grid(3000), stream=stream0)
        del buf272
        del convolution_19
        del primals_216
        del relu_13
        buf277 = empty((120, ), device='cuda', dtype=torch.float32)
        buf278 = buf277; del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf278, buf276, primals_217, 120, 25, grid=grid(120), stream=stream0)
        del primals_217
        buf282 = buf280[1]
        del buf280
        buf283 = reinterpret_tensor(buf276, (120, 25), (25, 1), 0); del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_64.run(relu_12, buf281, buf283, 3000, 126, grid=grid(3000), stream=stream0)
        buf284 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_65.run(buf283, buf284, 120, 25, grid=grid(120), stream=stream0)
        buf285 = reinterpret_tensor(buf283, (120, 25), (1, 120), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(relu_12, buf281, convolution_18, primals_213, buf285, 3000, 126, grid=grid(3000), stream=stream0)
        del buf281
        del convolution_18
        del primals_213
        del relu_12
        buf286 = empty((120, ), device='cuda', dtype=torch.float32)
        buf287 = buf286; del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf287, buf285, primals_214, 120, 25, grid=grid(120), stream=stream0)
        del primals_214
        buf291 = buf289[1]
        del buf289
        buf298 = buf296[1]
        del buf296
        buf299 = reinterpret_tensor(buf285, (120, 25), (25, 1), 0); del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_64.run(relu_11, buf297, buf299, 3000, 126, grid=grid(3000), stream=stream0)
        buf300 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_65.run(buf299, buf300, 120, 25, grid=grid(120), stream=stream0)
        buf301 = reinterpret_tensor(buf299, (120, 25), (1, 120), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(relu_11, buf297, convolution_16, primals_207, buf301, 3000, 126, grid=grid(3000), stream=stream0)
        del buf297
        del convolution_16
        del primals_207
        del relu_11
        buf302 = empty((120, ), device='cuda', dtype=torch.float32)
        buf303 = buf302; del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf303, buf301, primals_208, 120, 25, grid=grid(120), stream=stream0)
        del primals_208
        buf307 = buf305[1]
        del buf305
        buf308 = reinterpret_tensor(buf301, (120, 25), (25, 1), 0); del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_64.run(relu_10, buf306, buf308, 3000, 126, grid=grid(3000), stream=stream0)
        buf309 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_65.run(buf308, buf309, 120, 25, grid=grid(120), stream=stream0)
        buf310 = reinterpret_tensor(buf308, (120, 25), (1, 120), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(relu_10, buf306, convolution_15, primals_204, buf310, 3000, 126, grid=grid(3000), stream=stream0)
        del buf306
        del convolution_15
        del primals_204
        del relu_10
        buf311 = empty((120, ), device='cuda', dtype=torch.float32)
        buf312 = buf311; del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf312, buf310, primals_205, 120, 25, grid=grid(120), stream=stream0)
        del buf310
        del primals_205
        buf316 = buf314[1]
        del buf314
        buf320 = buf264; del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_68.run(buf320, buf290, buf315, primals_202, primals_44, 125440, grid=grid(125440), stream=stream0)
        del buf290
        del buf315
        del primals_202
        del primals_44
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf321 = aten.convolution_backward(buf320, relu_9, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf320
        del primals_43
        buf322 = buf321[0]
        buf323 = buf321[1]
        del buf321
        buf324 = empty((72, 25), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_69.run(relu_9, buf322, buf324, 1800, 126, grid=grid(1800), stream=stream0)
        buf325 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_70.run(buf324, buf325, 72, 25, grid=grid(72), stream=stream0)
        buf326 = reinterpret_tensor(buf324, (72, 25), (1, 72), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(relu_9, buf322, convolution_13, primals_198, buf326, 1800, 126, grid=grid(1800), stream=stream0)
        del convolution_13
        del primals_198
        buf327 = empty((72, ), device='cuda', dtype=torch.float32)
        buf328 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_72.run(buf328, buf326, primals_199, 72, 25, grid=grid(72), stream=stream0)
        del buf326
        buf329 = reinterpret_tensor(buf103, (4, 72, 28, 28), (56448, 1, 2016, 72), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_73.run(relu_9, buf322, primals_199, primals_41, buf329, 3136, 72, grid=grid(3136, 72), stream=stream0)
        del buf322
        del primals_199
        del primals_41
        del relu_9
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf330 = aten.convolution_backward(buf329, relu_8, primals_40, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 72, [True, True, False])
        del buf329
        del primals_40
        buf331 = buf330[0]
        buf332 = buf330[1]
        del buf330
        buf333 = empty((72, 98), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_8, buf331, buf333, 7056, 128, grid=grid(7056), stream=stream0)
        buf334 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf333, buf334, 72, 98, grid=grid(72), stream=stream0)
        buf335 = reinterpret_tensor(buf333, (72, 98), (1, 72), 0); del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_76.run(relu_8, buf331, convolution_12, primals_195, buf335, 7056, 128, grid=grid(7056), stream=stream0)
        del convolution_12
        del primals_195
        buf336 = empty((72, ), device='cuda', dtype=torch.float32)
        buf337 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_77.run(buf337, buf335, primals_196, 72, 98, grid=grid(72), stream=stream0)
        buf338 = empty_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_78.run(relu_8, buf331, primals_196, primals_38, buf338, 12544, 72, grid=grid(12544, 72), stream=stream0)
        del buf331
        del primals_196
        del primals_38
        del relu_8
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf339 = aten.convolution_backward(buf338, add_25, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_25
        del primals_37
        buf340 = buf339[0]
        buf341 = buf339[1]
        del buf339
        buf347 = empty((4, 24, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_79.run(buf340, primals_193, primals_35, buf347, 301056, grid=grid(301056), stream=stream0)
        del primals_35
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf348 = aten.convolution_backward(buf347, relu_7, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_34
        buf349 = buf348[0]
        buf356 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_78.run(relu_7, buf349, primals_190, primals_32, buf356, 12544, 72, grid=grid(12544, 72), stream=stream0)
        del primals_32
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf357 = aten.convolution_backward(buf356, relu_6, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False])
        del primals_31
        buf358 = buf357[0]
        buf365 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_78.run(relu_6, buf358, primals_187, primals_29, buf365, 12544, 72, grid=grid(12544, 72), stream=stream0)
        del primals_29
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf366 = aten.convolution_backward(buf365, add_18, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_18
        del primals_28
        buf367 = buf366[0]
        buf374 = buf347; del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_80.run(buf340, buf367, primals_184, primals_26, buf374, 301056, grid=grid(301056), stream=stream0)
        del primals_26
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf375 = aten.convolution_backward(buf374, relu_5, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf374
        del primals_25
        buf376 = buf375[0]
        buf383 = buf365; del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_78.run(relu_5, buf376, primals_181, primals_23, buf383, 12544, 72, grid=grid(12544, 72), stream=stream0)
        del primals_23
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf384 = aten.convolution_backward(buf383, relu_4, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False])
        del primals_22
        buf385 = buf384[0]
        buf392 = buf383; del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_78.run(relu_4, buf385, primals_178, primals_20, buf392, 12544, 72, grid=grid(12544, 72), stream=stream0)
        del primals_20
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf393 = aten.convolution_backward(buf392, add_11, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_11
        del buf392
        del primals_19
        buf394 = buf393[0]
        buf342 = empty_strided((24, 2), (1, 24), device='cuda', dtype=torch.float32)
        buf369 = empty_strided((24, 2), (1, 24), device='cuda', dtype=torch.float32)
        buf371 = empty_strided((24, 2), (1, 24), device='cuda', dtype=torch.float32)
        buf396 = empty_strided((24, 2), (1, 24), device='cuda', dtype=torch.float32)
        buf398 = empty_strided((24, 2), (1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_81.run(buf340, buf367, convolution_8, primals_183, buf394, convolution_5, primals_174, buf342, buf369, buf371, buf396, buf398, 48, 6272, grid=grid(48), stream=stream0)
        del convolution_5
        del convolution_8
        del primals_174
        del primals_183
        buf343 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_82.run(buf342, buf343, 24, 2, grid=grid(24), stream=stream0)
        del buf342
        buf344 = empty((24, 98), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_83.run(buf340, convolution_11, primals_192, buf344, 2352, 128, grid=grid(2352), stream=stream0)
        del convolution_11
        del primals_192
        buf345 = empty((24, ), device='cuda', dtype=torch.float32)
        buf346 = buf345; del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_84.run(buf346, buf344, primals_193, 24, 98, grid=grid(24), stream=stream0)
        del buf344
        del primals_193
        buf350 = buf348[1]
        del buf348
        buf351 = reinterpret_tensor(buf335, (72, 98), (98, 1), 0); del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_7, buf349, buf351, 7056, 128, grid=grid(7056), stream=stream0)
        buf352 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf351, buf352, 72, 98, grid=grid(72), stream=stream0)
        buf353 = reinterpret_tensor(buf351, (72, 98), (1, 72), 0); del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_85.run(relu_7, buf349, convolution_10, primals_189, buf353, 7056, 128, grid=grid(7056), stream=stream0)
        del buf349
        del convolution_10
        del primals_189
        del relu_7
        buf354 = empty((72, ), device='cuda', dtype=torch.float32)
        buf355 = buf354; del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_77.run(buf355, buf353, primals_190, 72, 98, grid=grid(72), stream=stream0)
        del primals_190
        buf359 = buf357[1]
        del buf357
        buf360 = reinterpret_tensor(buf353, (72, 98), (98, 1), 0); del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_6, buf358, buf360, 7056, 128, grid=grid(7056), stream=stream0)
        buf361 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf360, buf361, 72, 98, grid=grid(72), stream=stream0)
        buf362 = reinterpret_tensor(buf360, (72, 98), (1, 72), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_85.run(relu_6, buf358, convolution_9, primals_186, buf362, 7056, 128, grid=grid(7056), stream=stream0)
        del buf358
        del convolution_9
        del primals_186
        del relu_6
        buf363 = empty((72, ), device='cuda', dtype=torch.float32)
        buf364 = buf363; del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_77.run(buf364, buf362, primals_187, 72, 98, grid=grid(72), stream=stream0)
        del primals_187
        buf368 = buf366[1]
        del buf366
        buf370 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_82.run(buf369, buf370, 24, 2, grid=grid(24), stream=stream0)
        buf372 = empty((24, ), device='cuda', dtype=torch.float32)
        buf373 = buf372; del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_86.run(buf373, buf371, primals_184, 24, 2, grid=grid(24), stream=stream0)
        del primals_184
        buf377 = buf375[1]
        del buf375
        buf378 = reinterpret_tensor(buf362, (72, 98), (98, 1), 0); del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_5, buf376, buf378, 7056, 128, grid=grid(7056), stream=stream0)
        buf379 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf378, buf379, 72, 98, grid=grid(72), stream=stream0)
        buf380 = reinterpret_tensor(buf378, (72, 98), (1, 72), 0); del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_85.run(relu_5, buf376, convolution_7, primals_180, buf380, 7056, 128, grid=grid(7056), stream=stream0)
        del buf376
        del convolution_7
        del primals_180
        del relu_5
        buf381 = empty((72, ), device='cuda', dtype=torch.float32)
        buf382 = buf381; del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_77.run(buf382, buf380, primals_181, 72, 98, grid=grid(72), stream=stream0)
        del primals_181
        buf386 = buf384[1]
        del buf384
        buf387 = reinterpret_tensor(buf380, (72, 98), (98, 1), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_74.run(relu_4, buf385, buf387, 7056, 128, grid=grid(7056), stream=stream0)
        buf388 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_75.run(buf387, buf388, 72, 98, grid=grid(72), stream=stream0)
        buf389 = reinterpret_tensor(buf387, (72, 98), (1, 72), 0); del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_85.run(relu_4, buf385, convolution_6, primals_177, buf389, 7056, 128, grid=grid(7056), stream=stream0)
        del buf385
        del convolution_6
        del primals_177
        del relu_4
        buf390 = empty((72, ), device='cuda', dtype=torch.float32)
        buf391 = buf390; del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_77.run(buf391, buf389, primals_178, 72, 98, grid=grid(72), stream=stream0)
        del buf389
        del primals_178
        buf395 = buf393[1]
        del buf393
        buf397 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_82.run(buf396, buf397, 24, 2, grid=grid(24), stream=stream0)
        buf399 = empty((24, ), device='cuda', dtype=torch.float32)
        buf400 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_86.run(buf400, buf398, primals_175, 24, 2, grid=grid(24), stream=stream0)
        buf401 = buf340; del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_87.run(buf401, buf367, buf394, primals_175, primals_17, 301056, grid=grid(301056), stream=stream0)
        del buf367
        del buf394
        del primals_17
        del primals_175
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf402 = aten.convolution_backward(buf401, relu_3, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf401
        del primals_16
        buf403 = buf402[0]
        buf404 = buf402[1]
        del buf402
        buf405 = empty((48, 98), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_88.run(relu_3, buf403, buf405, 4704, 128, grid=grid(4704), stream=stream0)
        buf406 = reinterpret_tensor(buf398, (48, ), (1, ), 0); del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_89.run(buf405, buf406, 48, 98, grid=grid(48), stream=stream0)
        buf407 = reinterpret_tensor(buf405, (48, 98), (1, 48), 0); del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_90.run(relu_3, buf403, convolution_4, primals_171, buf407, 4704, 128, grid=grid(4704), stream=stream0)
        del convolution_4
        del primals_171
        buf408 = reinterpret_tensor(buf396, (48, ), (1, ), 0); del buf396  # reuse
        buf409 = buf408; del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_91.run(buf409, buf407, primals_172, 48, 98, grid=grid(48), stream=stream0)
        del buf407
        buf410 = empty_strided((4, 48, 56, 56), (150528, 1, 2688, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_92.run(relu_3, buf403, primals_172, primals_14, buf410, 12544, 48, grid=grid(12544, 48), stream=stream0)
        del buf403
        del primals_14
        del primals_172
        del relu_3
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf411 = aten.convolution_backward(buf410, relu_2, primals_13, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 48, [True, True, False])
        del buf410
        del primals_13
        buf412 = buf411[0]
        buf413 = buf411[1]
        del buf411
        buf414 = empty((48, 392), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_93.run(relu_2, buf412, buf414, 18816, 128, grid=grid(18816), stream=stream0)
        buf415 = reinterpret_tensor(buf371, (48, ), (1, ), 0); del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_94.run(buf414, buf415, 48, 392, grid=grid(48), stream=stream0)
        buf416 = reinterpret_tensor(buf414, (48, 392), (1, 48), 0); del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_95.run(relu_2, buf412, convolution_3, primals_168, buf416, 18816, 128, grid=grid(18816), stream=stream0)
        del convolution_3
        del primals_168
        buf417 = reinterpret_tensor(buf369, (48, ), (1, ), 0); del buf369  # reuse
        buf418 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_96.run(buf418, buf416, primals_169, 48, 392, grid=grid(48), stream=stream0)
        del buf416
        buf419 = empty_strided((4, 48, 112, 112), (602112, 1, 5376, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_97.run(relu_2, buf412, primals_169, primals_11, buf419, 50176, 48, grid=grid(50176, 48), stream=stream0)
        del buf412
        del primals_11
        del primals_169
        del relu_2
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf420 = aten.convolution_backward(buf419, add_5, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_5
        del buf419
        del primals_10
        buf421 = buf420[0]
        buf422 = buf420[1]
        del buf420
        buf423 = empty((16, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_98.run(buf421, buf423, 112, 7168, grid=grid(112), stream=stream0)
        buf424 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_99.run(buf423, buf424, 16, 7, grid=grid(16), stream=stream0)
        del buf423
        buf425 = empty((16, 392), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_100.run(buf421, convolution_2, primals_165, buf425, 6272, 128, grid=grid(6272), stream=stream0)
        del convolution_2
        del primals_165
        buf426 = empty((16, ), device='cuda', dtype=torch.float32)
        buf427 = buf426; del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_101.run(buf427, buf425, primals_166, 16, 392, grid=grid(16), stream=stream0)
        del buf425
        buf428 = buf421; del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_102.run(buf428, primals_166, primals_8, 802816, grid=grid(802816), stream=stream0)
        del primals_166
        del primals_8
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf429 = aten.convolution_backward(buf428, relu_1, primals_7, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf428
        del primals_7
        buf430 = buf429[0]
        buf431 = buf429[1]
        del buf429
        buf432 = empty((32, 392), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_103.run(relu_1, buf430, buf432, 12544, 128, grid=grid(12544), stream=stream0)
        buf433 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_104.run(buf432, buf433, 32, 392, grid=grid(32), stream=stream0)
        buf434 = reinterpret_tensor(buf432, (32, 392), (1, 32), 0); del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_105.run(relu_1, buf430, convolution_1, primals_162, buf434, 12544, 128, grid=grid(12544), stream=stream0)
        del convolution_1
        del primals_162
        buf435 = empty((32, ), device='cuda', dtype=torch.float32)
        buf436 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_106.run(buf436, buf434, primals_163, 32, 392, grid=grid(32), stream=stream0)
        buf437 = empty_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_107.run(relu_1, buf430, primals_163, primals_5, buf437, 50176, 32, grid=grid(50176, 32), stream=stream0)
        del buf430
        del primals_163
        del primals_5
        del relu_1
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf438 = aten.convolution_backward(buf437, relu, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del primals_4
        buf439 = buf438[0]
        buf440 = buf438[1]
        del buf438
        buf441 = reinterpret_tensor(buf434, (32, 392), (392, 1), 0); del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_103.run(relu, buf439, buf441, 12544, 128, grid=grid(12544), stream=stream0)
        buf442 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_104.run(buf441, buf442, 32, 392, grid=grid(32), stream=stream0)
        buf443 = reinterpret_tensor(buf441, (32, 392), (1, 32), 0); del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_105.run(relu, buf439, convolution, primals_159, buf443, 12544, 128, grid=grid(12544), stream=stream0)
        del convolution
        del primals_159
        buf444 = empty((32, ), device='cuda', dtype=torch.float32)
        buf445 = buf444; del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_106.run(buf445, buf443, primals_160, 32, 392, grid=grid(32), stream=stream0)
        del buf443
        buf446 = buf437; del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_107.run(relu, buf439, primals_160, primals_2, buf446, 50176, 32, grid=grid(50176, 32), stream=stream0)
        del buf439
        del primals_160
        del primals_2
        del relu
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf447 = aten.convolution_backward(buf446, primals_315, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf446
        del primals_1
        del primals_315
        buf448 = buf447[1]
        return (buf448, buf445, buf442, buf440, buf436, buf433, buf431, buf427, buf424, buf422, buf418, buf415, buf413, buf409, buf406, buf404, buf400, buf397, buf395, buf391, buf388, buf386, buf382, buf379, buf377, buf373, buf370, buf368, buf364, buf361, buf359, buf355, buf352, buf350, buf346, buf343, buf341, buf337, buf334, buf332, buf328, buf325, buf323, buf319, buf317, buf316, buf312, buf309, buf307, buf303, buf300, buf298, buf294, buf292, buf291, buf287, buf284, buf282, buf278, buf275, buf273, buf269, buf266, buf265, buf261, buf258, buf256, buf252, buf249, buf247, buf243, buf241, buf240, buf236, buf233, buf231, buf227, buf224, buf222, buf218, buf216, buf215, buf211, buf208, buf206, buf202, buf199, buf197, buf193, buf190, buf189, buf185, buf182, buf180, buf176, buf173, buf171, buf167, buf165, buf164, buf160, buf157, buf155, buf151, buf148, buf146, buf142, buf139, buf138, buf134, buf131, buf129, buf125, buf122, buf120, buf116, buf114, buf113, buf109, buf106, buf104, buf100, buf97, buf95, buf91, buf89, buf88, buf84, buf81, buf79, buf75, buf72, buf70, buf66, buf64, buf63, buf59, buf56, buf54, buf50, buf47, buf45, buf41, buf38, buf37, buf33, buf30, buf28, buf24, buf21, buf19, buf15, buf12, buf11, buf7, buf4, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), buf2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((48, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((48, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((40, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((96, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((576, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((192, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    primals_273 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    add_5 = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((4, 48, 112, 112), (602112, 1, 5376, 48), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((4, 48, 112, 112), (602112, 1, 5376, 48), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((4, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((4, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    add_11 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    add_18 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    add_25 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((4, 72, 28, 28), (56448, 1, 2016, 72), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((4, 72, 28, 28), (56448, 1, 2016, 72), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    add_31 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    add_38 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    add_45 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    add_51 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    add_58 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    add_65 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    relu_21 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cuda:0', dtype=torch.float32)
    add_71 = rand_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    relu_23 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cuda:0', dtype=torch.float32)
    add_78 = rand_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    relu_24 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((4, 576, 7, 7), (28224, 1, 4032, 576), device='cuda:0', dtype=torch.float32)
    relu_25 = rand_strided((4, 576, 7, 7), (28224, 1, 4032, 576), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    add_84 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    relu_26 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    relu_27 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    add_91 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    relu_28 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    relu_29 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    add_98 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    relu_31 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    add_105 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    relu_32 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    relu_33 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((4, 320, 7, 7), (15680, 1, 2240, 320), device='cuda:0', dtype=torch.float32)
    add_111 = rand_strided((4, 320, 7, 7), (15680, 1, 2240, 320), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((4, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda:0', dtype=torch.bool)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_159, primals_160, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, convolution, relu, convolution_1, relu_1, convolution_2, add_5, convolution_3, relu_2, convolution_4, relu_3, convolution_5, add_11, convolution_6, relu_4, convolution_7, relu_5, convolution_8, add_18, convolution_9, relu_6, convolution_10, relu_7, convolution_11, add_25, convolution_12, relu_8, convolution_13, relu_9, convolution_14, add_31, convolution_15, relu_10, convolution_16, relu_11, convolution_17, add_38, convolution_18, relu_12, convolution_19, relu_13, convolution_20, add_45, convolution_21, relu_14, convolution_22, relu_15, convolution_23, add_51, convolution_24, relu_16, convolution_25, relu_17, convolution_26, add_58, convolution_27, relu_18, convolution_28, relu_19, convolution_29, add_65, convolution_30, relu_20, convolution_31, relu_21, convolution_32, add_71, convolution_33, relu_22, convolution_34, relu_23, convolution_35, add_78, convolution_36, relu_24, convolution_37, relu_25, convolution_38, add_84, convolution_39, relu_26, convolution_40, relu_27, convolution_41, add_91, convolution_42, relu_28, convolution_43, relu_29, convolution_44, add_98, convolution_45, relu_30, convolution_46, relu_31, convolution_47, add_105, convolution_48, relu_32, convolution_49, relu_33, convolution_50, add_111, convolution_51, mean, permute_1, le, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mnasnet1_0', benchmark_compiled_module)
