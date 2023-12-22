
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


# kernel path: /tmp/torchinductor_youkaichao/eb/cebxvz4da6lcwqwr373toun5d6pifvqzeo2bansamqjkrj44lth2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]

triton_poi_fused_convolution_backward_hardswish_backward_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask)
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
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bn/cbnup4jegnm7ef6cfjwmpuutdvwhxavjh2xv2ydmzwmyatoeii5z.py
# Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_div_hardswish_backward_native_batch_norm_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_hardswish_backward_native_batch_norm_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1344
    x1 = (xindex // 1344)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp19 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1344*r2) + (131712*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + (x0 + (1344*(r2 // 49)) + (2688*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr2 + (x0 + (1344*r2) + (131712*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/av/cav7cagpjyowtezfwvwmpx53l64hjawhgyn4jawz3sbuneyphghs.py
# Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_div_hardswish_backward_native_batch_norm_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_hardswish_backward_native_batch_norm_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1344
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1344*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l5/cl5c4ojzuo57lkjmgyaep7kwmjqz7qwgvultnmr467cgti4dkazf.py
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
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1344
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1344*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bz/cbz4rlwfabev2oapa24hg3ymnokn47db2gxspx5kymqxqkthecar.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 526848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1344
    x2 = (xindex // 65856)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x0 + (1344*x2)), xmask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/72/c72kqgscmiludl2fdalfmzmoca3vscnntdpvvkizip65stok7uma.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 224
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (10976*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yv/cyvnqo66mpodcxltxckyp2ghinark3l3sxw4jej6ylrm6d2bqqdv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 224
    x1 = (xindex // 224)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (10976*(r2 // 49)) + (21952*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (224*r2) + (21952*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4m/c4m6r4fe42hyshy5vihwmk2a6ys46jrzmmqep7luggh3egzahnnj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_8', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/jw/cjwiw2dd6xv3m5s7676pg3s5novbhqlpwn6bjy4lecaipa3kie6y.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1792
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 224
    y1 = (yindex // 224)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (224*x2) + (10976*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i4/ci4h3rmxm4bgdjrjdezh7okiakarda7xy2hdnnqwabh7vnkaduvg.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_10', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8832
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1104
    x1 = (xindex // 1104)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1104*r2) + (54096*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/us/cusszq3aumnn6l5fmo5y2dh7dkvswi5bk5laipw7ggc2y2qv2ly6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1104
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1104*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/43/c43sblxgf3y4glorvecinhngdn2wdbtzydietrpoqxwwewcq2u7b.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward]

triton_poi_fused_hardswish_backward_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask)
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
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gq/cgqkjpgbickqlavmhp7hxesdfazscr7dmvv5uim7om3nddp5z7lf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f7/cf7gnuwsliwb5rfwxiv662bz3waf7ymfwsu5yn2xni6h2px3dtyg.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4416
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1104
    x1 = (xindex // 1104)
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp23 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1104*r2) + (108192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((49*x0) + (54096*(r2 // 49)) + (108192*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (1104*(r2 // 49)) + (2208*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (1104*(r2 // 49)) + (2208*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr4 + (x0 + (1104*r2) + (108192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/tp/ctp4ekzqyidqi7lf2uifwyjsflyhwkbvz46tvloe3flipjgfpom5.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1104
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1104*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fr/cfrv6oniowozmvqnha2ak6qbh4qaym4lsmmkphvbfarv5dqbdfub.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1104
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1104*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q2/cq26nlykj7mk7rxe6eg7kv4yvcuseuslor6t6hezgu64sgtuzojd.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 1104
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1104*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (49*x2) + (54096*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (1104*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (1104*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x2 + (1104*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2 + (1104*y3)), tmp35, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2x/c2x7quijtongpkobva7hj3c7ggcv2wvgv4iu6bofifo77f5ueplv.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4416
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1104
    x1 = (xindex // 1104)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1104*r2) + (108192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((49*x0) + (54096*(r2 // 49)) + (108192*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr2 + (x0 + (1104*r2) + (108192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/52/c52lbvht6dfdj76m7di2zrwhpbfukl6gfsog2dmiwu4lepexeq2p.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 1104
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1104*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (49*x2) + (54096*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (1104*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (1104*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uy/cuyapmqffu33v6wtn7yczwky6wt2j2hwkw4s47nlwizfpdttsncb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 184
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (9016*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uq/cuqa7v3bbtbl66ntt6rajlmtpi7imtvybmajtfgxclvp6bmsvekc.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 736
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 184
    x1 = (xindex // 184)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (9016*(r2 // 49)) + (18032*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (184*r2) + (18032*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4t/c4tjib5rkakj5l2yfj3icvaidsqiwnplf2rn7xxu6paknazj2y7g.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 184
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/t5/ct5vy4ait24arwt3zxhhvppbnwu26ndedkbvzmst4utb7276ztmw.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1472
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 184
    y1 = (yindex // 184)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (184*x2) + (9016*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/dk/cdkcuyiva7e33ztqjojemhuekqgpklkat3zg4eknxqfhb4zgpgtf.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_24', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5888
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 736
    x1 = (xindex // 736)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (736*r2) + (36064*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ot/cotazmw5jhsiyw2o5cjztmok55cg7zrjrxcef4ziw6fdzwpui5bz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 736
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (736*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oh/coh742w6nrp52zkv23kj5mjwbinijgwrw6xcgmadpjyvnyclnacr.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2944
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 736
    x1 = (xindex // 736)
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp23 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (736*r2) + (72128*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((49*x0) + (36064*(r2 // 49)) + (72128*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (736*(r2 // 49)) + (1472*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (736*(r2 // 49)) + (1472*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr4 + (x0 + (736*r2) + (72128*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/md/cmdtg6h2t4a3w7gwmb3xhvtt62ldcylt3newaa5jl6no5ocybfkq.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 736
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (736*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ol/colbuv4bx24q4s32ixuw5pby654rmhyksacvce2ycrvg5ovc76nf.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 736
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (736*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zf/czfbo6cr4tpcq4t6j5qfnvyvsqmrhdnrz3l65nbfz7u2ykzev5ec.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_29', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 736
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
    tmp0 = tl.load(in_ptr0 + (x2 + (736*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (49*x2) + (36064*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (736*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (736*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x2 + (736*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2 + (736*y3)), tmp35, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/as/cas6ftm6x652kae5elctsnq6t6qgrgzagzpsomwq6brlk7dsngti.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2944
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 736
    x1 = (xindex // 736)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (736*r2) + (72128*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((49*x0) + (36064*(r2 // 49)) + (72128*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr2 + (x0 + (736*r2) + (72128*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/v3/cv3xfcenwwf3tnp2zyvlokpb2mvyahdlkd2fufv6fmqjfu7nh7g2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 736
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
    tmp0 = tl.load(in_ptr0 + (x2 + (736*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (49*x2) + (36064*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (736*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (736*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/li/cliyn7sbf4sha5gb4lq7b6yrwn5lomp6e57ccmghlkccndj6jmqf.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 184
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (9016*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (9016*r2)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0 + (184*r3)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/24/c247m5f7azjpeqj276djg4vxwkqreqwbbc23rtiljmwopgl6cgff.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1472
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 184
    y1 = (yindex // 184)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (184*x2) + (9016*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/gk/cgkxujqypxb6od2kdkwptjkpefbmcf7bhqlqbiqffci3fmbghp2a.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 184
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (9016*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (9016*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (49*x0) + (9016*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (x0 + (184*r3)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ju/cju7uh2dpd2vcxsgbzidqp3d7posz3muoerj43vge3ht7roer5xe.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1472
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 184
    y1 = (yindex // 184)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (184*x2) + (9016*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ag/cag2durxsdt4ahzooc3ex6kamofo5fygg6gsweq324klxg5uvgfd.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 184
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (9016*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (9016*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (49*x0) + (9016*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (49*x0) + (9016*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr4 + (x0 + (184*r3)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = tmp18 * tmp19
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6h/c6hcydwap6fab5qvy43jekj2cyjzl4pe7bncvpkojs2b54rpys7i.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1472
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 184
    y1 = (yindex // 184)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0 + (184*x2) + (9016*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g5/cg5kca3rqimw4ke7b27q2i4vwkhrxftjtjrivxphqdthpjo6cvwf.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 72128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/il/cilhblgua4npx5grilqukvuatgxqpkzzgikbs4a6hpn2ljx7eqdo.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_39', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1472
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 184
    y1 = (yindex // 184)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (184*x2) + (9016*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sx/csxwe4fiocyjm4lmcf72ewh6zud5mvmcag6mzu3nw2s2esbyh6s6.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_40', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5760
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 720
    x1 = (xindex // 720)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (720*r2) + (35280*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xxzrivji36vwwf3tjdjo32xwsl4f3dcg7zii374qjaiyhzifas.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 720
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (720*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ry/cry7tbui53zyasnvck3niwmudgy5naxf2ac7p6tqo5pv63jtzdpm.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward]

triton_poi_fused_hardswish_backward_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_42', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask)
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
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ij/cijbk7gd66bk2vghcqojgrlgtg3fixvgeo5x2lxw2sj53vgozkjp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_43', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ty/ctyfdh6skgfuit3ujsahqnfucqb4tsdcihswwcja67kp4vtg7lxm.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2880
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 720
    x1 = (xindex // 720)
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp23 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (720*r2) + (70560*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((49*x0) + (35280*(r2 // 49)) + (70560*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (720*(r2 // 49)) + (1440*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (720*(r2 // 49)) + (1440*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr4 + (x0 + (720*r2) + (70560*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/by/cby337xezcnx7fzps6dhok5l6b22c2rqtid6zjma5pmzxp2q4lmu.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 720
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (720*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rb/crbdds2yboxtl2lcbubt4aaeycrj53glxxae33tjrnjyyxvu5grr.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 720
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (720*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/db/cdb37txaoymhxpjujbjfavpfgltybsfk2c6niunzyiu342booycj.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 720
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
    tmp0 = tl.load(in_ptr0 + (x2 + (720*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (49*x2) + (35280*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (720*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (720*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x2 + (720*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2 + (720*y3)), tmp35, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t5/ct5vhigctaxng4kuhk7f3eh75jg3sr5manhi2pukreozisrxaeqk.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9360
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
        tmp3 = tl.load(in_ptr0 + (x1 + (720*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x1) + (141120*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/hm/chmdawgbltdho3iu6ph64ledhmsaowpz2pk2ftq3yj3qd2g5sqwj.py
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
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 720
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


# kernel path: /tmp/torchinductor_youkaichao/x3/cx3mzrvzbqjgpz25xtfcsmd5syxrofxzc56svkbuxzmjzzy3tldu.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9360
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 720)
    x0 = xindex % 720
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (720*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x0) + (141120*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp3 / tmp6
        tmp10 = 0.5
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.where(tmp7, tmp12, tmp8)
        tmp14 = 0.0
        tmp15 = tl.where(tmp5, tmp14, tmp13)
        tmp16 = tl.load(in_ptr2 + (x0 + (720*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/eg/cegj2hmiggc4cocgfixfrp33u4gwpmhr763mxwvdowndhdhyqfcs.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 720
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (720*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pv/cpv6dsyx5skgegi274v7dvj4omb22os3ydfrpnszfhtz6lgisr6v.py
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
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 720
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
    tmp0 = tl.load(in_ptr0 + (x2 + (720*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (141120*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (720*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (720*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sk/csksq6kaeotwm4nxcuvezn6w3mkhfnkl5oijbx3chardha67l3yx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 120
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (23520*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nh/cnhmaaziagbt2c54mduygmuixilzrj6zbkzr3c7xzy4mzq7emsis.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1560
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (23520*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (120*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/w7/cw7u3p2wzbfamthkrsy5pwad5qqdch3tmcmoqbpqaw754jc6e4f5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 120
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


# kernel path: /tmp/torchinductor_youkaichao/i6/ci6b6vp3vfy46bbmax7wdwtgsfaefsdfnkhg6yg2553st6glggd7.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (120*x2) + (23520*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/ye/cyeg6xi5ybrvmtgvac3dyxuydeadfm5drmdppik6fxkzsldohjzt.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5760
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 2
    x1 = (xindex // 2) % 360
    x2 = (xindex // 720)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (98*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (360*r3) + (35280*x0) + (70560*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3l/c3lkd4satgw4mspgilh44aoeivcc2tttgxyj24hjims5k4cfusqk.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_58', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2880
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


# kernel path: /tmp/torchinductor_youkaichao/37/c37ysjpa3dtrgppkzil6qzzrhl4viux7lkj52y6cs3lcdniulrs3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 360
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (360*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uz/cuzdj3crxkaq236cxl4okio3qucv7c6v7mderlwuhseyi3jbvxqv.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4680
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
        tmp3 = tl.load(in_ptr0 + (x1 + (360*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x1) + (70560*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x1 + (360*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 * tmp9
        tmp11 = tl.load(in_ptr3 + (x1 + (360*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bx/cbxp3yyjc74rbc4vicdnflrglanhugj4ivipmcq6ka3am2qnkg7e.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 360
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


# kernel path: /tmp/torchinductor_youkaichao/ea/ceam5nx2a75enmwryzi7623s2klzofvfy3jdvihgbzka3vjtrrrj.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4680
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 360)
    x0 = xindex % 360
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (360*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x0) + (70560*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (360*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 * tmp9
        tmp11 = tl.load(in_ptr3 + (x0 + (360*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp22 = tl.load(in_ptr4 + (x0 + (360*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ic/ciczg3tj3wo33bnl3ch3il5uvnwjsm4puh4gcnixjwrbcrbgn5v3.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_63 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 360
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (360*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h3/ch32fm6mm2fnhnrjvoeywpw2oh6iyp4ye3qbhqaflu5lchbqxyc7.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_64', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 360
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
    tmp0 = tl.load(in_ptr0 + (x2 + (360*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (70560*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (360*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (360*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x2 + (360*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2 + (360*y3)), tmp35, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/il/cil3gn6pjfyqnh6unx2gtis64gg7ckxlujvex3f6leio6l35ent7.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4680
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
        tmp3 = tl.load(in_ptr0 + (x1 + (360*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x1) + (70560*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/nu/cnuindbyhqvtiefsearahm3qg55ccyykfsyu5c2woq6tt6jwwtwx.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4680
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 360)
    x0 = xindex % 360
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (360*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x0) + (70560*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp3 / tmp6
        tmp10 = 0.5
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.where(tmp7, tmp12, tmp8)
        tmp14 = 0.0
        tmp15 = tl.where(tmp5, tmp14, tmp13)
        tmp16 = tl.load(in_ptr2 + (x0 + (360*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/wl/cwl52ajvdqbyat6imbdlhnw2ewomycbqfgzomily2zs7wcvosxrm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 360
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
    tmp0 = tl.load(in_ptr0 + (x2 + (360*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (70560*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (360*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (360*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/so/csoi6lvhm4zldatvh2ub5hern4g6gfgq4mzvkdytu74qviuaiizz.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 120
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (23520*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (23520*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (120*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/sb/csb7zktnk3uffomhkeuilv2iylpnnqjqjh2o5ikarhqrfuinkzfv.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (120*x2) + (23520*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/gw/cgw2lozoqpev35o72t4pxgv5flquktjvucuplj7c4jdhtrpk2rke.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 120
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (23520*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (23520*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (196*x0) + (23520*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (120*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/6w/c6wrcty7s4tybg7lzghvcicyolqj3uisebjcdq2j6ri6kk63vtox.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (120*x2) + (23520*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/z5/cz5l47gkjdou7qouidctdivu3w6vewwvdpjytroo4wguvphzhdi2.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 120
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (23520*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (23520*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (196*x0) + (23520*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r1 + (196*x0) + (23520*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr4 + (x0 + (120*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/jz/cjzfyxthv63liel4qykwy4lyxupod3yc7pbpms3dj7stxq5wshpp.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_73', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0 + (120*x2) + (23520*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/3w/c3wdg5kiplhkztoblhn75vfkig5apkqnfijq3ukihdgywhtnuisx.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_74', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j6/cj6rlhf3koxmifbmi3mtlncnv5bsbc753bxub6fzjwg7yvtl35ae.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_75', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (120*x2) + (23520*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/7l/c7l5dzmzoqyuoxrol5z3dykhbqgq2m7ljat352olw4mrhmjf2ls4.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward]

triton_poi_fused_hardswish_backward_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_76', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask)
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
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ca/ccai2qv2zxilsvb3hwcbcyn2v26htpsk4gljs5moq7rtb2xqv4og.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_77', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/if/cif24a6w6lbcnhmdbmpkkwv7llbng6c5mu6wa5u7oqpnfjhlxecd.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (14112*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zg/czggp6t2okeljt4vhp2xtg32lyr7xzrdhb572u2yjymjyjy66jap.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_79', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 936
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (14112*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (72*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/dn/cdnmhtpaci5rzhupr7uidkjivhmmud3z46tcwly4g5e7nabzrtmd.py
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
    size_hints=[128, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 72
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


# kernel path: /tmp/torchinductor_youkaichao/4w/c4wi7aooxfpjoopqp4qsobu34bdhxljq6gisfbhuirxc7eszeson.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_81', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 72
    y1 = (yindex // 72)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (72*x2) + (14112*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/gs/cgsyyn6vwznzkaay2mzm5zkwbfinxstocb2ol6m6p5e63f2sme2h.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_82', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2808
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
        tmp3 = tl.load(in_ptr0 + (x1 + (216*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x1) + (42336*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/is/cistx7cdrhg43g6s7mixxhpz7v7sk5zvv2db2dkpa243er4rxvw2.py
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
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_83', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 216
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


# kernel path: /tmp/torchinductor_youkaichao/fn/cfncec5jtz3rrqofk64sqyjaea5vsdrr334ummp7buegnjktzbme.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_84', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2808
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 216)
    x0 = xindex % 216
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (216*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((196*x0) + (42336*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp3 / tmp6
        tmp10 = 0.5
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.where(tmp7, tmp12, tmp8)
        tmp14 = 0.0
        tmp15 = tl.where(tmp5, tmp14, tmp13)
        tmp16 = tl.load(in_ptr2 + (x0 + (216*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ms/cmszrcx3rflt7tlmikddxf5rjcpjw6ojzilxhznf2kqikqx4yx3x.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 216
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (216*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ia/ciagt56gefgkkjoeqlhhv326k45a2wg2ywiuu3yt6u4p2bsbgmfv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_86', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 216
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
    tmp0 = tl.load(in_ptr0 + (x2 + (216*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (42336*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (216*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (216*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nj/cnjujjp7wwoko7gtlk2uav7lfvc6ug6ubw3ucqjkiezcywh2sie4.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (14112*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (14112*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (72*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7s/c7sj5ull5534n4adgxuxad5vuxnnodmo5vdmtwixkgteeze3shz3.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 72
    y1 = (yindex // 72)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (72*x2) + (14112*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/zx/czxgf4ulbdulgqf374bvvu6opjtrkdbd7yhbf6dnox333f2z43mo.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_89 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (14112*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (14112*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (196*x0) + (14112*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (72*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7l/c7ls2pxez5c63u6fhkj5dzv7opbun75tvxac7rfbujflfozfb6jr.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_90', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 72
    y1 = (yindex // 72)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (72*x2) + (14112*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/pj/cpj2g2xjkticsnhzpwtqqt57gwivjoswlhymaqdcvit727w6vlvo.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_91 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (14112*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (14112*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (196*x0) + (14112*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r1 + (196*x0) + (14112*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr4 + (x0 + (72*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7n/c7n3ebfya4umvlvnemym2f5ek4wsktydhbnmyybwiqug5e3vpyvn.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_92 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_92', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 72
    y1 = (yindex // 72)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0 + (72*x2) + (14112*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/6s/c6sonzlb37c762le2x4pvuzxmeqf55f64qtckaeyaduwrkdqsx4k.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_93', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wy/cwyvb3bz5aidxkamdiml6n6s326mpjzbz7ak2jxaaxjcfy7uzibc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_94 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_94', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 72
    y1 = (yindex // 72)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (72*x2) + (14112*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ur/curwqmhf237azbzpyka7wo5h2rwz4blzf576jj6lhcqrw3xloi7x.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_95', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/f5/cf5ivisf2bljzfqkaya6wnumwaiuce5gujr2pjjzlxdodyz23req.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_96', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/tn/ctnnozttwd3qbb7ebiopbqjifqzcitzzmz7xjmqmctu54aobg3iz.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_97', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/qa/cqaly2h7p5cfhr6vozlob462oxyy3ksq5vms2meuqzqo5k2ampy4.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_98', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/d6/cd6jzm7yj23ihmczajjcugm6nkc24urct7kiyjixhvni54k5vzrx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_99', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/gx/cgxj64czlmwsdn6n2dv6t2pz4u5qp2evboonrphggjdzspvajj6f.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_100 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_100', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9800
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
        tmp0 = tl.load(in_ptr0 + (x1 + (200*r2) + (25600*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((784*x1) + (156800*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/27/c27gqo3tvnz5grqduf27ptofti7yxtvr3dgzn6s7esqieeygshzi.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_101 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_101', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 200
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


# kernel path: /tmp/torchinductor_youkaichao/cl/ccl3rfhpnzn5gvgxivkmsdeaqlaqnrycotpk6yfrsqavmfmqowwi.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_102 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_102', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9800
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 200
    x1 = (xindex // 200)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (200*r2) + (25600*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((784*x0) + (156800*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (200*r2) + (25600*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/3s/c3sititqvysut3dqyfuxkh7ua2at7so5msxzk5knry6mdfylovv5.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_103 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_103', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 200
    rnumel = 49
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/sz/cszs3iamtkbp7l5anno7rop6juuhe5ddnsnx6un2lbglzzkfkscf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_104 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_104', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 200
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
    tmp0 = tl.load(in_ptr0 + (x2 + (200*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (784*x2) + (156800*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (200*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mp/cmpxki3rx6cxaya5r55evilem6ecrhsmwudamftnvcghwo73u6sd.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_105 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_105', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/sc/csczlh527cbvt3ac5j554bxid2x3vwnhpgkn62umas3zlji7fa3j.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_106 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_106', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/kt/cktmrkrjrgadmqqdaktwhtahu5xeamu4jg7z3g4nfjdpjz2yupwp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_107 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_107', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/mq/cmqfkuucmb5idsbuo5kng6vh2vod75avr43j3ororzvcvhpqx5ai.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_108 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_108', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/jg/cjgyxisjnzldajvtoyjmpck5svlhnywqy6rn3kysbbq6uu3ozzwq.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_109 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_109', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/5y/c5y7dz2znrhi7udg7bdj4pdu5464dczukblcbjzstachladidmjh.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_110', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/di/cdijdk4roq5r5vfbkax7vz7ufs5cuymisjnavxcrsumxklb5dts6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_111 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_111', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/cb/ccbqrg2qettpjcr2ddtna47jshdxiozbl2fru36tdoldjig32uhd.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward]

triton_poi_fused_hardswish_backward_112 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_112', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask)
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
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/io/cionxz3gdohoon424indjlv6osyhkfpeay4tttf6abvdsupt3b3n.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_113 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_113', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y4/cy4xqzgtbt3fchjrzi4brsx442mou5f2mxhhizh6dqvhmtvkh2ok.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_114 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_114', 'mutated_arg_names': []}
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
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (120*r2) + (15360*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((784*x1) + (94080*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x1 + (120*((r2 + (128*x0)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (120*((r2 + (128*x0)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp7 = tmp5 * tmp6
        tmp9 = 784.0
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
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/op/copgkxpyezn3nay4l4vciq5whdz2kxcvo3yvublpsse435uceoys.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_115 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_115', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/7s/c7sfrdy2ijmarbzjxw22ixu2g6f7n7pfeogetrv3ikhqtyf3mqjd.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_116 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_116', 'mutated_arg_names': []}
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
    tmp20 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (120*r2) + (15360*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((784*x0) + (94080*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (120*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (120*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr4 + (x0 + (120*r2) + (15360*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp7 = tmp5 * tmp6
        tmp9 = 784.0
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
        tmp22 = tmp18 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ad/caddjz5xzfma4etulkzxakfc6quwukzgt5jir2u2hvmb5jqggium.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_117 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_117', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/tp/ctpzcqgefqg2ro3bvpdlfjdjyqnlzatkglxyo4qqgxtv3gpqw7g5.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_118 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_118', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp5 = tl.load(in_ptr1 + (y0 + (784*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (120*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (120*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x2 + (120*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp9 = 784.0
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
    tmp23 = 0.00015943877551020407
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
    tl.store(in_out_ptr0 + (x2 + (120*y3)), tmp35, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/py/cpyubu6dtg5totutreo6oxjzphrcjyuxkkvifn4dig6qnvrhv4kj.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_119 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_119', 'mutated_arg_names': []}
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
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (120*r2) + (15360*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((784*x1) + (94080*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ma/cmagnonwmu22hf2sxposansokruypvuph663a4m6xgcb2o3awox7.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_120 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_120', 'mutated_arg_names': []}
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
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (120*r2) + (15360*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((784*x0) + (94080*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (120*r2) + (15360*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/u5/cu5egtc5t74pztteuqdpdcfoqz7ftmcishwcw4245fk75bmoru3n.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_121 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_121', 'mutated_arg_names': []},
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
    tmp5 = tl.load(in_ptr1 + (y0 + (784*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (120*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (120*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jf/cjfiypab7orqfcwibviynv3i6uslj6254hv6gsthft5ml52c4nkx.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_122 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_122', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/3x/c3xyvu4ijv5hwapevpw63kcdesojmhgn2b65eyomxqbpyqeu5qod.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_123 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_123', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/gw/cgwcd2xhv2mddhathfb2x6tnhya6ubjbboakzckc4rwc3g4hld4b.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_124 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_124', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/4a/c4atyrp7yi7pra4543eyat7sqaaaijzvj7cf7r6nrrupqwdabw5b.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_125 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_125', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr2 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (40*x2) + (31360*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oo/coojdmf2226tyt2it2brh25wbzykc4hizhs4q5ym6vptryzvwmmn.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_126 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_126', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 6272
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
        r1 = rindex % 784
        r2 = (rindex // 784)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (784*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (784*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r1 + (784*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr4 + (x0 + (40*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/65/c65sg2puqupdklepmz7ezu2yqftsk4phafd77qysf57dv46fxekz.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_127 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_127', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr2 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0 + (40*x2) + (31360*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xh/cxhvqlbxal73p3oxvv3pdqxosh3gcndkytpflinjoqxhp3awxuhn.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_128 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_128', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p3/cp3563cmrm2pplgf7nrid5wotrdn4fwjszezzqrooud5rl6mh5hv.py
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
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_129', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr0 + (y0 + (40*x2) + (31360*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3q/c3q262fwn32f27mbcc4kbad7yffchloet42xje3ahpnfpc7s5yz7.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward]

triton_poi_fused_hardswish_backward_130 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_130', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask)
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
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g7/cg7wkvjlaz2guog4p7bgjqzetqybmojezsbp6kf4bukupxw76yqp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_131 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_131', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (8*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pc/cpcpom7xaxmyfxb2fneqtcgpezciynzo2dohvnktrv23uvgd5w5n.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_132 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_132', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 23520
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
        tmp0 = tl.load(in_ptr0 + (x1 + (120*r2) + (15360*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((3136*x1) + (376320*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/xh/cxh22ea7b65z2euylal3w3nzt4poxskcqoedkcs2cycep5iehfj6.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_133 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_133', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 120
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


# kernel path: /tmp/torchinductor_youkaichao/5j/c5jxxqyepcqfcpjolavfk4nxtopslu6ozwk63qgfxzvu7ify5jya.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_134 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_134', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 23520
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 120
    x1 = (xindex // 120)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (120*r2) + (15360*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((3136*x0) + (376320*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (120*r2) + (15360*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/lr/clr3ltr2tuynu7tybijjqxgbdq742ojekwsjcqhjijfix3hxlrqj.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_135 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_135', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 120
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
        tmp0 = tl.load(in_ptr0 + (x0 + (120*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3a/c3acuyuttwhladbnyimpld4huiyhnhdleg4p3ciurhsrrxbld4gg.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_136 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_136', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 120
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
    tmp0 = tl.load(in_ptr0 + (x2 + (120*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (3136*x2) + (376320*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (120*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (120*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kc/ckclukskanzjhnnpec2hw5ixbuqvhfcwxlqhppgtctgdjm77zwzb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_137 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_137', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/hz/chzvj5ttdlyzhlcah3qm7kglrgufyj2yutwww2tcahsdikwmjszg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_138 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_138', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/w7/cw7bcuy6dhutaskoa4hbofd3j7iu7fz3emdw7gkjkls5qvzopry5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_139 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_139', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/lv/clvzyapymyi6vtfpenjetvmdelwuwvt6rlzjaa4npcmhdq7rtgir.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_140 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_140', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/q7/cq7b5k3h3uf6ghtyynesphxlvyisfftncfbyjvs7nlo2esugoqfs.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_141 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_141', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/t4/ct4fasuqtntnri6qhudqjvfs6zd3f3t7gmoi63umpz2kfq5q4hyx.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_142 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_142', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
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
        tmp0 = tl.load(in_ptr0 + (x1 + (48*r2) + (6144*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((3136*x1) + (150528*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/xm/cxmiqc4esvwpkez2nkasnurp6eccpmfbdbilqxg43psws5oajiqn.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_143 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_143', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 48
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


# kernel path: /tmp/torchinductor_youkaichao/c3/cc3dfxq5wuxbemmori4426y2kylxcay4rlr5fpqo7rgb2alilsvj.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_144 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_144', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 48
    x1 = (xindex // 48)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (48*r2) + (6144*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((3136*x0) + (150528*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (48*r2) + (6144*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bw/cbw7us6gmztmo6tgd3nmm4dis4g76qq7e5iawvirlvekrl6quvva.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_145 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_145', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 48
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
        tmp0 = tl.load(in_ptr0 + (x0 + (48*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cj/ccjrw33l5uyvamytijdpvxx2vbea2a3qw6g3q3h4s3l2uhjliqlx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_146 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_146', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
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
    tmp5 = tl.load(in_ptr1 + (y0 + (3136*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (48*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (48*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n3/cn3rziu4ahqpm7bwmqgkj4t7wdv2srhoyzl6av2iuzknyd2nmfbn.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_147 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_147', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/cn/ccnuhpfklqgbvq67r3wv2zyfzpugnocbiw27walj4sssoypv2fha.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_148 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_148', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/je/cje4uv272jj2waulaepp3isc43gqmvytueg2g74nlojn4rsqwyns.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_149 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_149', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (24*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o4/co4da7pelauwaxpw4efkup75happ7fnk6ajhybk35rwfbueomchn.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_150 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_150', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (24*r2) + (150528*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k4/ck4vws7szmg7nw2q6dm2i5nlpq23docpqrcrtijtwq4t5f77w5qy.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_151 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_151', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (24*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dc/cdciht733fxqwxlzsooedzwnpmu6d46x5pyyuamjnpbvvn3ogj7g.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_152 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_152', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr4 + (x0 + (24*r2) + (150528*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5kcz6j5k632mgo7ojqm4uq2loiqejfpw6vhggsgnhhvlwvvqmx.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_153 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_153', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr1 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (24*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (3136*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ux/cuxej5wj4cii2udxmq4g5ufajhg62ldsvul3jb6t5qrxv27o7v2l.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_154 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_154', 'mutated_arg_names': []}
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
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((3136*x1) + (200704*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/2j/c2jgpevc6qhoq35wbfaem2xgpncy46433fdjetbcv6k46q3n7lxu.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_155 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_155', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ir/cir3huo55wir4bko7qfbqntutqpqcav5gnuhp3j2zr47byuvuhog.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_156 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_156', 'mutated_arg_names': []}
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
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((3136*x0) + (200704*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/s5/cs5ynlzy2isdshsirbpmqpevxaunpgouazsdxid5xzguwuwddign.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_157 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_157', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/fh/cfhr2xjgcxu23fgy5tsavya6tx6qmapg7vsq3wkd4xkj6ucqtwop.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_158 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_158', 'mutated_arg_names': []},
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
    tmp5 = tl.load(in_ptr1 + (y0 + (3136*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nx/cnxr4pqfvw2772uebfae4aj6rz7jhl6nyfelpjpygvug4htv2hmc.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_159 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_159', 'mutated_arg_names': []}
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
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((12544*x1) + (802816*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4b/c4b6c7326en6xrdgzqgk4srkgn3uyyb5zcdi2wdmugwwprdj3i56.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_160 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_160', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/j5/cj542pdjhaffxa3d6dcejlrc2flep6m27wtuddh3cfrqca43ceod.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_161 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_161', 'mutated_arg_names': []}
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
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((12544*x0) + (802816*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/2v/c2vbn3zrcjqxs5ok2yh52gihmorbxsoywlhqwf3orhnsc6whl5mv.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_162 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_162', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/s4/cs4waltaabez3nrvhhovdp6e2bbfj5uvtnqzebusae33ecau3otd.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_163 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_163', 'mutated_arg_names': []},
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
    tmp5 = tl.load(in_ptr1 + (y0 + (12544*x2) + (802816*y1)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fq/cfqf66uahg47kvyqmxhvfg6tvtmy6nu4jhr5jcm7csi3f5oebxgr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_164 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_164', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/g3/cg3ab7swmzf5mbss5ohzc6dyeidzry4ykj65hrcnadg4tomfz5p2.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_165 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_165', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/yj/cyjofp7ljzpcvjdgl5vja5tfinne4c4ritbwk7y7aptwsfbzcryt.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_166 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_166', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/u5/cu56y2dabugu2w2fhvxdce6nsdch3cwtz2vszvfzjdemtsnatsps.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_167 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_167', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/d3/cd3x7plg6rnjsjevi7lejzo6ynjilwikxe7d556boz72oycwerbh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_168 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_168', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/7s/c7saolexsoq4zavqf7w6rme2hhqpvkuidzvrxl3czkpkjnc4hgo5.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_169 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_169', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/yo/cyol6ixaioedklbsyqkojcv6rnk62otz63vceiehur2nins72etx.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_170 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_170', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/us/cus7mhiikcx2qpqlonv3fv5db7qutqae3p73lz477ti3z2mv7x6m.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_171 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_171', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/lm/clm5cg7yhghblovvtocftiv6uxnc4jddcn7h3ekuu2irpxiudrmy.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_172 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_172', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/xa/cxaeskvqe67gkmyiynwgveslh45thpnyafwdfjqornj6tloym7on.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_173 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_173', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/3s/c3shtu4urm7mj5ryp4g4yij3vdfeumstxqgxm2faemfryl77colt.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_174 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_174', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 208
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x0)
        tmp1 = tl.full([1, 1], 100352, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((12544*x1) + (200704*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((12544*x1) + (200704*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp11 = tl.load(in_ptr2 + (x1 + (16*((r2 + (7720*x0)) % 100352))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr3 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp5 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/62/c62prjh6qjpfvdnukz535dhg5rlmftevqs2nwbzhtvvvrtos6yj4.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_175 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_175', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mq/cmqjg5x23a65x6bcdqvppdmy2jm3qimhdgffujyvnxik6ar5jm66.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_176 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_176', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (16*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 9.964923469387754e-06
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + (x2 + (12544*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2n/c2nmzhw2vh7o3ul6qjwbknbriffcwfequxencnpatz25z5d5jmj5.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_add_hardswish_backward_native_batch_norm_backward_177 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardswish_backward_native_batch_norm_backward_177', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 208
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp32 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
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
        tmp11 = tl.load(in_ptr3 + ((12544*x1) + (200704*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp3 / tmp6
        tmp14 = 0.5
        tmp15 = tmp13 + tmp14
        tmp16 = tmp12 * tmp15
        tmp17 = tl.where(tmp7, tmp16, tmp12)
        tmp18 = 0.0
        tmp19 = tl.where(tmp5, tmp18, tmp17)
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
        tmp25 = tl.load(in_ptr4 + (x1 + (16*((r2 + (7720*x0)) % 100352))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr5 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tmp25 - tmp26
        tmp28 = tmp19 * tmp27
        tmp29 = tl.full(tmp28.shape, 0, tmp28.dtype)
        tmp30 = tl.where(tmp2, tmp28, tmp29)
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(rmask & xmask, tmp33, _tmp32)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, xmask)
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ho/choviucrwhg7465malowru65buwsbhgzhunm25kopcczusbirf72.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_hardswish_backward_native_batch_norm_backward_178 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_hardswish_backward_native_batch_norm_backward_178', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (16*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0 + (16*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp0 / tmp3
    tmp11 = 0.5
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 * tmp12
    tmp14 = tl.where(tmp4, tmp13, tmp9)
    tmp15 = 0.0
    tmp16 = tl.where(tmp2, tmp15, tmp14)
    tmp19 = tmp17 - tmp18
    tmp21 = 9.964923469387754e-06
    tmp22 = tmp20 * tmp21
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp19 * tmp25
    tmp27 = tmp16 - tmp26
    tmp29 = tmp28 * tmp21
    tmp30 = tmp27 - tmp29
    tmp32 = tmp23 * tmp31
    tmp33 = tmp30 * tmp32
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp33, xmask & ymask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_198, primals_200, primals_201, primals_202, primals_203, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_214, primals_215, primals_216, primals_217, primals_219, primals_221, primals_222, primals_223, primals_224, primals_226, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_248, primals_250, primals_251, primals_252, primals_253, primals_255, primals_257, primals_258, primals_259, primals_260, primals_262, primals_264, primals_265, primals_266, primals_267, primals_269, primals_271, primals_272, primals_273, primals_274, primals_276, primals_278, primals_279, primals_280, primals_281, primals_283, primals_285, primals_286, primals_287, primals_288, primals_290, primals_292, primals_293, primals_294, primals_295, primals_297, primals_299, primals_300, primals_301, primals_302, primals_304, primals_306, primals_307, primals_308, primals_309, primals_311, primals_313, primals_314, primals_315, primals_316, primals_318, primals_320, primals_321, primals_322, primals_323, primals_325, primals_327, primals_328, primals_329, primals_330, primals_332, primals_334, primals_335, primals_336, primals_598, convolution, squeeze_1, clone, div, convolution_1, squeeze_4, clone_1, div_1, convolution_2, squeeze_7, add_17, convolution_3, squeeze_10, clone_2, div_2, convolution_4, squeeze_13, add_29, convolution_5, squeeze_16, clone_3, div_3, convolution_6, squeeze_19, clone_4, div_4, convolution_7, squeeze_22, add_46, convolution_8, squeeze_25, clone_5, div_5, convolution_9, squeeze_28, clone_6, div_6, convolution_10, squeeze_31, add_64, convolution_11, squeeze_34, clone_7, div_7, convolution_12, squeeze_37, clone_8, div_8, convolution_13, squeeze_40, add_82, convolution_14, squeeze_43, clone_9, div_9, convolution_15, squeeze_46, clone_10, div_10, convolution_16, squeeze_49, add_100, convolution_17, squeeze_52, clone_11, div_11, convolution_18, squeeze_55, clone_12, div_12, mean, convolution_19, div_13, div_14, mul_147, convolution_21, squeeze_58, add_119, convolution_22, squeeze_61, clone_14, div_15, convolution_23, squeeze_64, clone_15, div_16, mean_1, convolution_24, div_17, div_18, mul_172, convolution_26, squeeze_67, add_139, convolution_27, squeeze_70, clone_17, div_19, convolution_28, squeeze_73, clone_18, div_20, mean_2, convolution_29, div_21, div_22, mul_197, convolution_31, squeeze_76, add_159, convolution_32, squeeze_79, clone_20, div_23, convolution_33, squeeze_82, clone_21, div_24, mean_3, convolution_34, div_25, div_26, mul_222, convolution_36, squeeze_85, add_179, convolution_37, squeeze_88, clone_23, div_27, convolution_38, squeeze_91, clone_24, div_28, mean_4, convolution_39, div_29, div_30, mul_247, convolution_41, squeeze_94, add_199, convolution_42, squeeze_97, clone_26, div_31, convolution_43, squeeze_100, clone_27, div_32, convolution_44, squeeze_103, add_216, convolution_45, squeeze_106, clone_28, div_33, convolution_46, squeeze_109, clone_29, div_34, convolution_47, squeeze_112, add_234, convolution_48, squeeze_115, clone_30, div_35, convolution_49, squeeze_118, clone_31, div_36, convolution_50, squeeze_121, add_252, convolution_51, squeeze_124, clone_32, div_37, convolution_52, squeeze_127, clone_33, div_38, convolution_53, squeeze_130, add_270, convolution_54, squeeze_133, clone_34, div_39, convolution_55, squeeze_136, clone_35, div_40, convolution_56, squeeze_139, add_288, convolution_57, squeeze_142, clone_36, div_41, convolution_58, squeeze_145, clone_37, div_42, mean_5, convolution_59, div_43, div_44, mul_387, convolution_61, squeeze_148, add_307, convolution_62, squeeze_151, clone_39, div_45, convolution_63, squeeze_154, clone_40, div_46, mean_6, convolution_64, div_47, div_48, mul_412, convolution_66, squeeze_157, add_327, convolution_67, squeeze_160, clone_42, div_49, convolution_68, squeeze_163, clone_43, div_50, mean_7, convolution_69, div_51, div_52, mul_437, convolution_71, squeeze_166, add_347, convolution_72, squeeze_169, clone_45, div_53, convolution_73, squeeze_172, clone_46, div_54, mean_8, convolution_74, div_55, div_56, mul_462, convolution_76, squeeze_175, add_367, convolution_77, squeeze_178, clone_48, div_57, convolution_78, squeeze_181, clone_49, div_58, mean_9, convolution_79, div_59, div_60, mul_487, convolution_81, squeeze_184, add_387, convolution_82, squeeze_187, clone_51, div_61, convolution_83, squeeze_190, clone_52, div_62, mean_10, convolution_84, div_63, div_64, mul_512, convolution_86, squeeze_193, add_407, convolution_87, squeeze_196, clone_54, div_65, convolution_88, squeeze_199, clone_55, div_66, mean_11, convolution_89, div_67, div_68, mul_537, convolution_91, squeeze_202, add_426, convolution_92, squeeze_205, clone_57, div_69, convolution_93, squeeze_208, clone_58, div_70, mean_12, convolution_94, div_71, div_72, mul_562, convolution_96, squeeze_211, add_446, convolution_97, squeeze_214, clone_60, div_73, convolution_98, squeeze_217, clone_61, div_74, mean_13, convolution_99, div_75, div_76, mul_587, convolution_101, squeeze_220, add_466, convolution_102, squeeze_223, clone_63, div_77, convolution_103, squeeze_226, clone_64, div_78, mean_14, convolution_104, div_79, div_80, mul_612, convolution_106, squeeze_229, add_486, convolution_107, squeeze_232, clone_66, div_81, convolution_108, squeeze_235, clone_67, div_82, mean_15, convolution_109, div_83, div_84, mul_637, convolution_111, squeeze_238, add_506, convolution_112, squeeze_241, clone_69, div_85, convolution_113, squeeze_244, clone_70, div_86, mean_16, convolution_114, div_87, div_88, mul_662, convolution_116, squeeze_247, add_526, convolution_117, squeeze_250, clone_72, div_89, convolution_118, squeeze_253, clone_73, div_90, mean_17, convolution_119, div_91, div_92, mul_687, convolution_121, squeeze_256, add_545, convolution_122, squeeze_259, clone_75, mean_18, convolution_123, view_1, permute_1, unsqueeze_350, unsqueeze_362, bitwise_and, unsqueeze_374, unsqueeze_386, unsqueeze_398, bitwise_and_1, unsqueeze_410, unsqueeze_422, unsqueeze_434, bitwise_and_2, unsqueeze_446, unsqueeze_458, unsqueeze_470, bitwise_and_3, unsqueeze_482, unsqueeze_494, unsqueeze_506, bitwise_and_4, unsqueeze_518, unsqueeze_530, unsqueeze_542, bitwise_and_5, unsqueeze_554, unsqueeze_566, unsqueeze_578, bitwise_and_6, unsqueeze_590, unsqueeze_602, unsqueeze_614, bitwise_and_7, unsqueeze_626, unsqueeze_638, unsqueeze_650, bitwise_and_8, unsqueeze_662, unsqueeze_674, unsqueeze_686, bitwise_and_9, unsqueeze_698, unsqueeze_710, unsqueeze_722, bitwise_and_10, unsqueeze_734, unsqueeze_746, unsqueeze_758, bitwise_and_11, unsqueeze_770, unsqueeze_782, unsqueeze_794, bitwise_and_12, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, unsqueeze_914, unsqueeze_926, unsqueeze_938, unsqueeze_950, unsqueeze_962, unsqueeze_974, unsqueeze_986, unsqueeze_998, unsqueeze_1010, bitwise_and_13, unsqueeze_1022, unsqueeze_1034, unsqueeze_1046, bitwise_and_14, unsqueeze_1058, unsqueeze_1070, unsqueeze_1082, bitwise_and_15, unsqueeze_1094, unsqueeze_1106, unsqueeze_1118, bitwise_and_16, unsqueeze_1130, unsqueeze_1142, unsqueeze_1154, bitwise_and_17, unsqueeze_1166, unsqueeze_1178, unsqueeze_1190, unsqueeze_1202, unsqueeze_1214, unsqueeze_1226, unsqueeze_1238, unsqueeze_1250, unsqueeze_1262, unsqueeze_1274, unsqueeze_1286, unsqueeze_1298, unsqueeze_1310, unsqueeze_1322, unsqueeze_1334, unsqueeze_1346, unsqueeze_1358, unsqueeze_1370, unsqueeze_1382, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (16, ), (1, ))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_9, (16, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_15, (24, ), (1, ))
    assert_size_stride(primals_17, (48, ), (1, ))
    assert_size_stride(primals_19, (48, ), (1, ))
    assert_size_stride(primals_21, (24, ), (1, ))
    assert_size_stride(primals_23, (48, ), (1, ))
    assert_size_stride(primals_25, (48, ), (1, ))
    assert_size_stride(primals_27, (24, ), (1, ))
    assert_size_stride(primals_29, (48, ), (1, ))
    assert_size_stride(primals_31, (48, ), (1, ))
    assert_size_stride(primals_33, (24, ), (1, ))
    assert_size_stride(primals_35, (120, ), (1, ))
    assert_size_stride(primals_37, (120, ), (1, ))
    assert_size_stride(primals_39, (40, ), (1, ))
    assert_size_stride(primals_41, (120, ), (1, ))
    assert_size_stride(primals_43, (120, ), (1, ))
    assert_size_stride(primals_45, (40, ), (1, ))
    assert_size_stride(primals_47, (120, ), (1, ))
    assert_size_stride(primals_49, (120, ), (1, ))
    assert_size_stride(primals_51, (40, ), (1, ))
    assert_size_stride(primals_53, (120, ), (1, ))
    assert_size_stride(primals_55, (120, ), (1, ))
    assert_size_stride(primals_57, (40, ), (1, ))
    assert_size_stride(primals_59, (120, ), (1, ))
    assert_size_stride(primals_61, (120, ), (1, ))
    assert_size_stride(primals_63, (40, ), (1, ))
    assert_size_stride(primals_65, (200, ), (1, ))
    assert_size_stride(primals_67, (200, ), (1, ))
    assert_size_stride(primals_69, (72, ), (1, ))
    assert_size_stride(primals_71, (216, ), (1, ))
    assert_size_stride(primals_73, (216, ), (1, ))
    assert_size_stride(primals_75, (72, ), (1, ))
    assert_size_stride(primals_77, (216, ), (1, ))
    assert_size_stride(primals_79, (216, ), (1, ))
    assert_size_stride(primals_81, (72, ), (1, ))
    assert_size_stride(primals_83, (216, ), (1, ))
    assert_size_stride(primals_85, (216, ), (1, ))
    assert_size_stride(primals_87, (72, ), (1, ))
    assert_size_stride(primals_89, (216, ), (1, ))
    assert_size_stride(primals_91, (216, ), (1, ))
    assert_size_stride(primals_93, (72, ), (1, ))
    assert_size_stride(primals_95, (360, ), (1, ))
    assert_size_stride(primals_97, (360, ), (1, ))
    assert_size_stride(primals_99, (120, ), (1, ))
    assert_size_stride(primals_101, (360, ), (1, ))
    assert_size_stride(primals_103, (360, ), (1, ))
    assert_size_stride(primals_105, (120, ), (1, ))
    assert_size_stride(primals_107, (360, ), (1, ))
    assert_size_stride(primals_109, (360, ), (1, ))
    assert_size_stride(primals_111, (120, ), (1, ))
    assert_size_stride(primals_113, (360, ), (1, ))
    assert_size_stride(primals_115, (360, ), (1, ))
    assert_size_stride(primals_117, (120, ), (1, ))
    assert_size_stride(primals_119, (360, ), (1, ))
    assert_size_stride(primals_121, (360, ), (1, ))
    assert_size_stride(primals_123, (120, ), (1, ))
    assert_size_stride(primals_125, (360, ), (1, ))
    assert_size_stride(primals_127, (360, ), (1, ))
    assert_size_stride(primals_129, (120, ), (1, ))
    assert_size_stride(primals_131, (720, ), (1, ))
    assert_size_stride(primals_133, (720, ), (1, ))
    assert_size_stride(primals_135, (184, ), (1, ))
    assert_size_stride(primals_137, (736, ), (1, ))
    assert_size_stride(primals_139, (736, ), (1, ))
    assert_size_stride(primals_141, (184, ), (1, ))
    assert_size_stride(primals_143, (736, ), (1, ))
    assert_size_stride(primals_145, (736, ), (1, ))
    assert_size_stride(primals_147, (184, ), (1, ))
    assert_size_stride(primals_149, (736, ), (1, ))
    assert_size_stride(primals_151, (736, ), (1, ))
    assert_size_stride(primals_153, (184, ), (1, ))
    assert_size_stride(primals_155, (736, ), (1, ))
    assert_size_stride(primals_157, (736, ), (1, ))
    assert_size_stride(primals_159, (184, ), (1, ))
    assert_size_stride(primals_161, (736, ), (1, ))
    assert_size_stride(primals_163, (736, ), (1, ))
    assert_size_stride(primals_165, (184, ), (1, ))
    assert_size_stride(primals_167, (1104, ), (1, ))
    assert_size_stride(primals_169, (1104, ), (1, ))
    assert_size_stride(primals_171, (224, ), (1, ))
    assert_size_stride(primals_173, (1344, ), (1, ))
    assert_size_stride(primals_177, (16, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_178, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_179, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_180, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_181, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_182, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_183, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_184, (24, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_185, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_186, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_187, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_188, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_189, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_190, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_191, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_192, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_193, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_194, (120, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_195, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_196, (8, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_198, (120, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_200, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_201, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_202, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_203, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_205, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_207, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_208, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_209, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_210, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_212, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_214, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_215, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_216, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_217, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_219, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_221, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_222, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_223, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_224, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_226, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_228, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_229, (200, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_230, (200, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_231, (72, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_232, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_233, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_234, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(primals_235, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_236, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_237, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(primals_238, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_239, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_240, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(primals_241, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_242, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_243, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(primals_244, (360, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_245, (360, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_246, (24, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_248, (360, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_250, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_251, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_252, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_253, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_255, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_257, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_258, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_259, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_260, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_262, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_264, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_265, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_266, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_267, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_269, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_271, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_272, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_273, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_274, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_276, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_278, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_279, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_280, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_281, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_283, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_285, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_286, (720, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_287, (720, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_288, (32, 720, 1, 1), (720, 1, 1, 1))
    assert_size_stride(primals_290, (720, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_292, (184, 720, 1, 1), (720, 1, 1, 1))
    assert_size_stride(primals_293, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_294, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_295, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_297, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_299, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_300, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_301, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_302, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_304, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_306, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_307, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_308, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_309, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_311, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_313, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_314, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_315, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_316, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_318, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_320, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_321, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_322, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_323, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_325, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_327, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_328, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_329, (1104, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_330, (48, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(primals_332, (1104, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_334, (224, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(primals_335, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_336, (1984, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_598, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_1, (16, ), (1, ))
    assert_size_stride(clone, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(div, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_1, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_4, (16, ), (1, ))
    assert_size_stride(clone_1, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(div_1, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_7, (16, ), (1, ))
    assert_size_stride(add_17, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_3, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_10, (16, ), (1, ))
    assert_size_stride(clone_2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(div_2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_4, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_13, (16, ), (1, ))
    assert_size_stride(add_29, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_5, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_16, (64, ), (1, ))
    assert_size_stride(clone_3, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(div_3, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(convolution_6, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_19, (64, ), (1, ))
    assert_size_stride(clone_4, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(div_4, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_7, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_22, (24, ), (1, ))
    assert_size_stride(add_46, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_8, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(squeeze_25, (48, ), (1, ))
    assert_size_stride(clone_5, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(div_5, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(convolution_9, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(squeeze_28, (48, ), (1, ))
    assert_size_stride(clone_6, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(div_6, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(convolution_10, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_31, (24, ), (1, ))
    assert_size_stride(add_64, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_11, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(squeeze_34, (48, ), (1, ))
    assert_size_stride(clone_7, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(div_7, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(convolution_12, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(squeeze_37, (48, ), (1, ))
    assert_size_stride(clone_8, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(div_8, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(convolution_13, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_40, (24, ), (1, ))
    assert_size_stride(add_82, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_14, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(squeeze_43, (48, ), (1, ))
    assert_size_stride(clone_9, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(div_9, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(convolution_15, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(squeeze_46, (48, ), (1, ))
    assert_size_stride(clone_10, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(div_10, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(convolution_16, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_49, (24, ), (1, ))
    assert_size_stride(add_100, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_17, (8, 120, 56, 56), (376320, 1, 6720, 120))
    assert_size_stride(squeeze_52, (120, ), (1, ))
    assert_size_stride(clone_11, (8, 120, 56, 56), (376320, 1, 6720, 120))
    assert_size_stride(div_11, (8, 120, 56, 56), (376320, 1, 6720, 120))
    assert_size_stride(convolution_18, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_55, (120, ), (1, ))
    assert_size_stride(clone_12, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(div_12, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(convolution_19, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(div_13, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(div_14, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_147, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_21, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_58, (40, ), (1, ))
    assert_size_stride(add_119, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_22, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_61, (120, ), (1, ))
    assert_size_stride(clone_14, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(div_15, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_23, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_64, (120, ), (1, ))
    assert_size_stride(clone_15, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(div_16, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean_1, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(convolution_24, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(div_17, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(div_18, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_172, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_26, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_67, (40, ), (1, ))
    assert_size_stride(add_139, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_27, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_70, (120, ), (1, ))
    assert_size_stride(clone_17, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(div_19, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_28, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_73, (120, ), (1, ))
    assert_size_stride(clone_18, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(div_20, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean_2, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(convolution_29, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(div_21, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(div_22, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_197, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_31, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_76, (40, ), (1, ))
    assert_size_stride(add_159, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_32, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_79, (120, ), (1, ))
    assert_size_stride(clone_20, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(div_23, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_33, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_82, (120, ), (1, ))
    assert_size_stride(clone_21, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(div_24, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean_3, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(convolution_34, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(div_25, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(div_26, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_222, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_36, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_85, (40, ), (1, ))
    assert_size_stride(add_179, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_37, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_88, (120, ), (1, ))
    assert_size_stride(clone_23, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(div_27, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_38, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_91, (120, ), (1, ))
    assert_size_stride(clone_24, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(div_28, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean_4, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(convolution_39, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(div_29, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(div_30, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_247, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_41, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_94, (40, ), (1, ))
    assert_size_stride(add_199, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_42, (8, 200, 28, 28), (156800, 1, 5600, 200))
    assert_size_stride(squeeze_97, (200, ), (1, ))
    assert_size_stride(clone_26, (8, 200, 28, 28), (156800, 1, 5600, 200))
    assert_size_stride(div_31, (8, 200, 28, 28), (156800, 1, 5600, 200))
    assert_size_stride(convolution_43, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(squeeze_100, (200, ), (1, ))
    assert_size_stride(clone_27, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(div_32, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(convolution_44, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(squeeze_103, (72, ), (1, ))
    assert_size_stride(add_216, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(convolution_45, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(squeeze_106, (216, ), (1, ))
    assert_size_stride(clone_28, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(div_33, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(convolution_46, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(squeeze_109, (216, ), (1, ))
    assert_size_stride(clone_29, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(div_34, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(convolution_47, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(squeeze_112, (72, ), (1, ))
    assert_size_stride(add_234, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(convolution_48, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(squeeze_115, (216, ), (1, ))
    assert_size_stride(clone_30, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(div_35, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(convolution_49, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(squeeze_118, (216, ), (1, ))
    assert_size_stride(clone_31, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(div_36, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(convolution_50, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(squeeze_121, (72, ), (1, ))
    assert_size_stride(add_252, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(convolution_51, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(squeeze_124, (216, ), (1, ))
    assert_size_stride(clone_32, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(div_37, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(convolution_52, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(squeeze_127, (216, ), (1, ))
    assert_size_stride(clone_33, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(div_38, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(convolution_53, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(squeeze_130, (72, ), (1, ))
    assert_size_stride(add_270, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(convolution_54, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(squeeze_133, (216, ), (1, ))
    assert_size_stride(clone_34, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(div_39, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(convolution_55, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(squeeze_136, (216, ), (1, ))
    assert_size_stride(clone_35, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(div_40, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(convolution_56, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(squeeze_139, (72, ), (1, ))
    assert_size_stride(add_288, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(convolution_57, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_142, (360, ), (1, ))
    assert_size_stride(clone_36, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_41, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_58, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_145, (360, ), (1, ))
    assert_size_stride(clone_37, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_42, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(mean_5, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(convolution_59, (8, 24, 1, 1), (24, 1, 24, 24))
    assert_size_stride(div_43, (8, 24, 1, 1), (24, 1, 24, 24))
    assert_size_stride(div_44, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(mul_387, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_61, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(squeeze_148, (120, ), (1, ))
    assert_size_stride(add_307, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(convolution_62, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_151, (360, ), (1, ))
    assert_size_stride(clone_39, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_45, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_63, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_154, (360, ), (1, ))
    assert_size_stride(clone_40, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_46, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(mean_6, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(convolution_64, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_47, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_48, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(mul_412, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_66, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(squeeze_157, (120, ), (1, ))
    assert_size_stride(add_327, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(convolution_67, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_160, (360, ), (1, ))
    assert_size_stride(clone_42, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_49, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_68, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_163, (360, ), (1, ))
    assert_size_stride(clone_43, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_50, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(mean_7, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(convolution_69, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_51, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_52, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(mul_437, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_71, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(squeeze_166, (120, ), (1, ))
    assert_size_stride(add_347, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(convolution_72, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_169, (360, ), (1, ))
    assert_size_stride(clone_45, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_53, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_73, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_172, (360, ), (1, ))
    assert_size_stride(clone_46, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_54, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(mean_8, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(convolution_74, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_55, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_56, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(mul_462, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_76, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(squeeze_175, (120, ), (1, ))
    assert_size_stride(add_367, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(convolution_77, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_178, (360, ), (1, ))
    assert_size_stride(clone_48, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_57, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_78, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_181, (360, ), (1, ))
    assert_size_stride(clone_49, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_58, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(mean_9, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(convolution_79, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_59, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_60, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(mul_487, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_81, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(squeeze_184, (120, ), (1, ))
    assert_size_stride(add_387, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(convolution_82, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_187, (360, ), (1, ))
    assert_size_stride(clone_51, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_61, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_83, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_190, (360, ), (1, ))
    assert_size_stride(clone_52, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_62, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(mean_10, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(convolution_84, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_63, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_64, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(mul_512, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_86, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(squeeze_193, (120, ), (1, ))
    assert_size_stride(add_407, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(convolution_87, (8, 720, 14, 14), (141120, 1, 10080, 720))
    assert_size_stride(squeeze_196, (720, ), (1, ))
    assert_size_stride(clone_54, (8, 720, 14, 14), (141120, 1, 10080, 720))
    assert_size_stride(div_65, (8, 720, 14, 14), (141120, 1, 10080, 720))
    assert_size_stride(convolution_88, (8, 720, 7, 7), (35280, 1, 5040, 720))
    assert_size_stride(squeeze_199, (720, ), (1, ))
    assert_size_stride(clone_55, (8, 720, 7, 7), (35280, 1, 5040, 720))
    assert_size_stride(div_66, (8, 720, 7, 7), (35280, 1, 5040, 720))
    assert_size_stride(mean_11, (8, 720, 1, 1), (720, 1, 720, 720))
    assert_size_stride(convolution_89, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_67, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_68, (8, 720, 1, 1), (720, 1, 720, 720))
    assert_size_stride(mul_537, (8, 720, 7, 7), (35280, 1, 5040, 720))
    assert_size_stride(convolution_91, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(squeeze_202, (184, ), (1, ))
    assert_size_stride(add_426, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(convolution_92, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_205, (736, ), (1, ))
    assert_size_stride(clone_57, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_69, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_93, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_208, (736, ), (1, ))
    assert_size_stride(clone_58, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_70, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(mean_12, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(convolution_94, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_71, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_72, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(mul_562, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_96, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(squeeze_211, (184, ), (1, ))
    assert_size_stride(add_446, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(convolution_97, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_214, (736, ), (1, ))
    assert_size_stride(clone_60, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_73, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_98, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_217, (736, ), (1, ))
    assert_size_stride(clone_61, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_74, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(mean_13, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(convolution_99, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_75, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_76, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(mul_587, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_101, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(squeeze_220, (184, ), (1, ))
    assert_size_stride(add_466, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(convolution_102, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_223, (736, ), (1, ))
    assert_size_stride(clone_63, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_77, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_103, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_226, (736, ), (1, ))
    assert_size_stride(clone_64, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_78, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(mean_14, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(convolution_104, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_79, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_80, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(mul_612, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_106, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(squeeze_229, (184, ), (1, ))
    assert_size_stride(add_486, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(convolution_107, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_232, (736, ), (1, ))
    assert_size_stride(clone_66, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_81, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_108, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_235, (736, ), (1, ))
    assert_size_stride(clone_67, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_82, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(mean_15, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(convolution_109, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_83, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_84, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(mul_637, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_111, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(squeeze_238, (184, ), (1, ))
    assert_size_stride(add_506, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(convolution_112, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_241, (736, ), (1, ))
    assert_size_stride(clone_69, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_85, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_113, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_244, (736, ), (1, ))
    assert_size_stride(clone_70, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_86, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(mean_16, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(convolution_114, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_87, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_88, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(mul_662, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_116, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(squeeze_247, (184, ), (1, ))
    assert_size_stride(add_526, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(convolution_117, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    assert_size_stride(squeeze_250, (1104, ), (1, ))
    assert_size_stride(clone_72, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    assert_size_stride(div_89, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    assert_size_stride(convolution_118, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    assert_size_stride(squeeze_253, (1104, ), (1, ))
    assert_size_stride(clone_73, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    assert_size_stride(div_90, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    assert_size_stride(mean_17, (8, 1104, 1, 1), (1104, 1, 1104, 1104))
    assert_size_stride(convolution_119, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_91, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_92, (8, 1104, 1, 1), (1104, 1, 1104, 1104))
    assert_size_stride(mul_687, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    assert_size_stride(convolution_121, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(squeeze_256, (224, ), (1, ))
    assert_size_stride(add_545, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_122, (8, 1344, 7, 7), (65856, 1, 9408, 1344))
    assert_size_stride(squeeze_259, (1344, ), (1, ))
    assert_size_stride(clone_75, (8, 1344, 7, 7), (65856, 1, 9408, 1344))
    assert_size_stride(mean_18, (8, 1344, 1, 1), (1344, 1, 1344, 1344))
    assert_size_stride(convolution_123, (8, 1984, 1, 1), (1984, 1, 1, 1))
    assert_size_stride(view_1, (8, 1984), (1984, 1))
    assert_size_stride(permute_1, (1000, 1984), (1984, 1))
    assert_size_stride(unsqueeze_350, (1, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(unsqueeze_362, (1, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(bitwise_and, (8, 1104, 1, 1), (1104, 1, 1104, 1104))
    assert_size_stride(unsqueeze_374, (1, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(unsqueeze_386, (1, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(unsqueeze_398, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(bitwise_and_1, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(unsqueeze_410, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_422, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_434, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(bitwise_and_2, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(unsqueeze_446, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_458, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_470, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(bitwise_and_3, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(unsqueeze_482, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_494, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_506, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(bitwise_and_4, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(unsqueeze_518, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_530, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_542, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(bitwise_and_5, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(unsqueeze_554, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_566, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_578, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(bitwise_and_6, (8, 720, 1, 1), (720, 1, 720, 720))
    assert_size_stride(unsqueeze_590, (1, 720, 1, 1), (720, 1, 1, 1))
    assert_size_stride(unsqueeze_602, (1, 720, 1, 1), (720, 1, 1, 1))
    assert_size_stride(unsqueeze_614, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(bitwise_and_7, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(unsqueeze_626, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_638, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_650, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(bitwise_and_8, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(unsqueeze_662, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_674, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_686, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(bitwise_and_9, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(unsqueeze_698, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_710, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_722, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(bitwise_and_10, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(unsqueeze_734, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_746, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_758, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(bitwise_and_11, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(unsqueeze_770, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_782, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_794, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(bitwise_and_12, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(unsqueeze_806, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_818, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_830, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_842, (1, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(unsqueeze_854, (1, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(unsqueeze_866, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_878, (1, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(unsqueeze_890, (1, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(unsqueeze_902, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_914, (1, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(unsqueeze_926, (1, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(unsqueeze_938, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_950, (1, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(unsqueeze_962, (1, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(unsqueeze_974, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_986, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_998, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_1010, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(bitwise_and_13, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(unsqueeze_1022, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1034, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1046, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(bitwise_and_14, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(unsqueeze_1058, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1070, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1082, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(bitwise_and_15, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(unsqueeze_1094, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1106, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1118, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(bitwise_and_16, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(unsqueeze_1130, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1142, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1154, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(bitwise_and_17, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(unsqueeze_1166, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1178, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1190, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_1202, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(unsqueeze_1214, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(unsqueeze_1226, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_1238, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(unsqueeze_1250, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(unsqueeze_1262, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_1274, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(unsqueeze_1286, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(unsqueeze_1298, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_1310, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1322, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1334, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_1346, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_1358, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_1370, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_1382, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 1984), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 1984), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), view_1, out=buf1)
        del view_1
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = reinterpret_tensor(buf0, (8, 1984, 1, 1), (1984, 1, 1, 1), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_1.run(buf3, convolution_123, 15872, grid=grid(15872), stream=stream0)
        del convolution_123
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
        buf4 = aten.convolution_backward(buf3, mean_18, primals_336, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf3
        del mean_18
        del primals_336
        buf5 = buf4[0]
        buf6 = buf4[1]
        del buf4
        buf7 = empty_strided((1344, 4), (1, 1344), device='cuda', dtype=torch.float32)
        buf9 = empty_strided((1344, 4), (1, 1344), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_div_hardswish_backward_native_batch_norm_backward_2.run(clone_75, buf5, convolution_122, unsqueeze_350, buf7, buf9, 5376, 98, grid=grid(5376), stream=stream0)
        buf8 = empty((1344, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_3.run(buf7, buf8, 1344, 4, grid=grid(1344), stream=stream0)
        del buf7
        buf10 = empty((1344, ), device='cuda', dtype=torch.float32)
        buf11 = empty((1344, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4.run(buf9, squeeze_259, buf10, buf11, 1344, 4, grid=grid(1344), stream=stream0)
        del buf9
        buf12 = empty_strided((8, 1344, 7, 7), (65856, 1, 9408, 1344), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_5.run(clone_75, buf5, convolution_122, unsqueeze_350, buf10, squeeze_259, buf8, primals_173, buf12, 526848, grid=grid(526848), stream=stream0)
        del buf10
        del buf5
        del clone_75
        del convolution_122
        del primals_173
        del squeeze_259
        del unsqueeze_350
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf13 = aten.convolution_backward(buf12, add_545, primals_335, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_545
        del buf12
        del primals_335
        buf14 = buf13[0]
        buf15 = buf13[1]
        del buf13
        buf16 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_6.run(buf14, buf16, 224, 392, grid=grid(224), stream=stream0)
        buf17 = empty_strided((224, 4), (1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_7.run(buf14, convolution_121, unsqueeze_362, buf17, 896, 98, grid=grid(896), stream=stream0)
        buf18 = empty((224, ), device='cuda', dtype=torch.float32)
        buf19 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_8.run(buf17, squeeze_256, buf18, buf19, 224, 4, grid=grid(224), stream=stream0)
        del buf17
        buf20 = buf14; del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_9.run(buf20, convolution_121, unsqueeze_362, buf18, squeeze_256, buf16, primals_171, 1792, 49, grid=grid(1792, 49), stream=stream0)
        del buf18
        del convolution_121
        del primals_171
        del squeeze_256
        del unsqueeze_362
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf21 = aten.convolution_backward(buf20, mul_687, primals_334, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf20
        del mul_687
        del primals_334
        buf22 = buf21[0]
        buf23 = buf21[1]
        del buf21
        buf24 = empty_strided((8, 1104, 1, 1), (1104, 1, 8832, 8832), device='cuda', dtype=torch.float32)
        buf25 = reinterpret_tensor(buf24, (8, 1104, 1, 1), (1104, 1, 1104, 1104), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_10.run(buf25, buf22, div_90, bitwise_and, 8832, 49, grid=grid(8832), stream=stream0)
        del bitwise_and
        del div_90
        buf26 = empty((1104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_11.run(buf25, buf26, 1104, 8, grid=grid(1104), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf27 = aten.convolution_backward(buf25, div_91, primals_332, [1104], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf25
        del div_91
        del primals_332
        buf28 = buf27[0]
        buf29 = buf27[1]
        del buf27
        buf30 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_12.run(buf30, convolution_119, 384, grid=grid(384), stream=stream0)
        del convolution_119
        buf31 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_13.run(buf30, buf31, 48, 8, grid=grid(48), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf32 = aten.convolution_backward(buf30, mean_17, primals_330, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf30
        del mean_17
        del primals_330
        buf33 = buf32[0]
        buf34 = buf32[1]
        del buf32
        buf35 = empty_strided((1104, 4), (1, 1104), device='cuda', dtype=torch.float32)
        buf37 = empty_strided((1104, 4), (1, 1104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_14.run(clone_73, buf22, div_92, buf33, convolution_118, unsqueeze_374, buf35, buf37, 4416, 98, grid=grid(4416), stream=stream0)
        buf36 = empty((1104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_15.run(buf35, buf36, 1104, 4, grid=grid(1104), stream=stream0)
        buf38 = empty((1104, ), device='cuda', dtype=torch.float32)
        buf40 = empty((1104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_16.run(buf37, squeeze_253, buf38, buf40, 1104, 4, grid=grid(1104), stream=stream0)
        buf39 = empty_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda', dtype=torch.float32)
        buf41 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_17.run(buf41, clone_73, buf22, div_92, buf33, convolution_118, unsqueeze_374, buf38, squeeze_253, buf36, primals_169, 392, 1104, grid=grid(392, 1104), stream=stream0)
        del buf22
        del buf33
        del clone_73
        del convolution_118
        del div_92
        del primals_169
        del squeeze_253
        del unsqueeze_374
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf42 = aten.convolution_backward(buf41, div_89, primals_329, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1104, [True, True, False])
        del div_89
        del primals_329
        buf43 = buf42[0]
        buf44 = buf42[1]
        del buf42
        buf45 = buf37; del buf37  # reuse
        buf47 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_18.run(clone_72, buf43, convolution_117, unsqueeze_386, buf45, buf47, 4416, 98, grid=grid(4416), stream=stream0)
        buf46 = buf38; del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_15.run(buf45, buf46, 1104, 4, grid=grid(1104), stream=stream0)
        del buf45
        buf48 = empty((1104, ), device='cuda', dtype=torch.float32)
        buf49 = empty((1104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_16.run(buf47, squeeze_250, buf48, buf49, 1104, 4, grid=grid(1104), stream=stream0)
        del buf47
        buf50 = buf41; del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_19.run(clone_72, buf43, convolution_117, unsqueeze_386, buf48, squeeze_250, buf46, primals_167, buf50, 392, 1104, grid=grid(392, 1104), stream=stream0)
        del buf43
        del buf48
        del clone_72
        del convolution_117
        del primals_167
        del squeeze_250
        del unsqueeze_386
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf51 = aten.convolution_backward(buf50, add_526, primals_328, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_526
        del buf50
        del primals_328
        buf52 = buf51[0]
        buf53 = buf51[1]
        del buf51
        buf54 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_20.run(buf52, buf54, 184, 392, grid=grid(184), stream=stream0)
        buf55 = empty_strided((184, 4), (1, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_21.run(buf52, convolution_116, unsqueeze_398, buf55, 736, 98, grid=grid(736), stream=stream0)
        buf56 = empty((184, ), device='cuda', dtype=torch.float32)
        buf57 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_22.run(buf55, squeeze_247, buf56, buf57, 184, 4, grid=grid(184), stream=stream0)
        buf58 = empty((8, 184, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_23.run(buf52, convolution_116, unsqueeze_398, buf56, squeeze_247, buf54, primals_165, buf58, 1472, 49, grid=grid(1472, 49), stream=stream0)
        del convolution_116
        del primals_165
        del squeeze_247
        del unsqueeze_398
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf59 = aten.convolution_backward(buf58, mul_662, primals_327, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_662
        del primals_327
        buf60 = buf59[0]
        buf61 = buf59[1]
        del buf59
        buf62 = empty_strided((8, 736, 1, 1), (736, 1, 5888, 5888), device='cuda', dtype=torch.float32)
        buf63 = reinterpret_tensor(buf62, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_24.run(buf63, buf60, div_86, bitwise_and_1, 5888, 49, grid=grid(5888), stream=stream0)
        del bitwise_and_1
        del div_86
        buf64 = reinterpret_tensor(buf55, (736, ), (1, ), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_25.run(buf63, buf64, 736, 8, grid=grid(736), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf65 = aten.convolution_backward(buf63, div_87, primals_325, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf63
        del div_87
        del primals_325
        buf66 = buf65[0]
        buf67 = buf65[1]
        del buf65
        buf68 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_12.run(buf68, convolution_114, 384, grid=grid(384), stream=stream0)
        del convolution_114
        buf69 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_13.run(buf68, buf69, 48, 8, grid=grid(48), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf70 = aten.convolution_backward(buf68, mean_16, primals_323, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf68
        del mean_16
        del primals_323
        buf71 = buf70[0]
        buf72 = buf70[1]
        del buf70
        buf73 = empty_strided((736, 4), (1, 736), device='cuda', dtype=torch.float32)
        buf75 = empty_strided((736, 4), (1, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_26.run(clone_70, buf60, div_88, buf71, convolution_113, unsqueeze_410, buf73, buf75, 2944, 98, grid=grid(2944), stream=stream0)
        buf74 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_27.run(buf73, buf74, 736, 4, grid=grid(736), stream=stream0)
        buf76 = empty((736, ), device='cuda', dtype=torch.float32)
        buf78 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_28.run(buf75, squeeze_244, buf76, buf78, 736, 4, grid=grid(736), stream=stream0)
        buf77 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        buf79 = buf77; del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_29.run(buf79, clone_70, buf60, div_88, buf71, convolution_113, unsqueeze_410, buf76, squeeze_244, buf74, primals_163, 392, 736, grid=grid(392, 736), stream=stream0)
        del buf60
        del clone_70
        del convolution_113
        del div_88
        del primals_163
        del squeeze_244
        del unsqueeze_410
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf80 = aten.convolution_backward(buf79, div_85, primals_322, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False])
        del div_85
        del primals_322
        buf81 = buf80[0]
        buf82 = buf80[1]
        del buf80
        buf83 = buf75; del buf75  # reuse
        buf85 = buf73; del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_30.run(clone_69, buf81, convolution_112, unsqueeze_422, buf83, buf85, 2944, 98, grid=grid(2944), stream=stream0)
        buf84 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_27.run(buf83, buf84, 736, 4, grid=grid(736), stream=stream0)
        buf86 = empty((736, ), device='cuda', dtype=torch.float32)
        buf87 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_28.run(buf85, squeeze_241, buf86, buf87, 736, 4, grid=grid(736), stream=stream0)
        buf88 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_31.run(clone_69, buf81, convolution_112, unsqueeze_422, buf86, squeeze_241, buf84, primals_161, buf88, 392, 736, grid=grid(392, 736), stream=stream0)
        del buf81
        del clone_69
        del convolution_112
        del primals_161
        del squeeze_241
        del unsqueeze_422
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf89 = aten.convolution_backward(buf88, add_506, primals_321, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_506
        del primals_321
        buf90 = buf89[0]
        buf91 = buf89[1]
        del buf89
        buf92 = buf56; del buf56  # reuse
        buf93 = empty((184, ), device='cuda', dtype=torch.float32)
        buf94 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_32.run(buf52, buf90, convolution_111, unsqueeze_434, squeeze_238, buf92, buf93, buf94, 184, 392, grid=grid(184), stream=stream0)
        buf95 = buf58; del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_33.run(buf52, buf90, convolution_111, unsqueeze_434, buf93, squeeze_238, buf92, primals_159, buf95, 1472, 49, grid=grid(1472, 49), stream=stream0)
        del convolution_111
        del primals_159
        del squeeze_238
        del unsqueeze_434
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf96 = aten.convolution_backward(buf95, mul_637, primals_320, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_637
        del primals_320
        buf97 = buf96[0]
        buf98 = buf96[1]
        del buf96
        buf99 = reinterpret_tensor(buf71, (8, 736, 1, 1), (736, 1, 5888, 5888), 0); del buf71  # reuse
        buf100 = reinterpret_tensor(buf99, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_24.run(buf100, buf97, div_82, bitwise_and_2, 5888, 49, grid=grid(5888), stream=stream0)
        del bitwise_and_2
        del div_82
        buf101 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_25.run(buf100, buf101, 736, 8, grid=grid(736), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf102 = aten.convolution_backward(buf100, div_83, primals_318, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf100
        del div_83
        del primals_318
        buf103 = buf102[0]
        buf104 = buf102[1]
        del buf102
        buf105 = buf103; del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_12.run(buf105, convolution_109, 384, grid=grid(384), stream=stream0)
        del convolution_109
        buf106 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_13.run(buf105, buf106, 48, 8, grid=grid(48), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf107 = aten.convolution_backward(buf105, mean_15, primals_316, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf105
        del mean_15
        del primals_316
        buf108 = buf107[0]
        buf109 = buf107[1]
        del buf107
        buf110 = buf85; del buf85  # reuse
        buf112 = buf83; del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_26.run(clone_67, buf97, div_84, buf108, convolution_108, unsqueeze_446, buf110, buf112, 2944, 98, grid=grid(2944), stream=stream0)
        buf111 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_27.run(buf110, buf111, 736, 4, grid=grid(736), stream=stream0)
        buf113 = empty((736, ), device='cuda', dtype=torch.float32)
        buf115 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_28.run(buf112, squeeze_235, buf113, buf115, 736, 4, grid=grid(736), stream=stream0)
        buf114 = buf88; del buf88  # reuse
        buf116 = buf114; del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_29.run(buf116, clone_67, buf97, div_84, buf108, convolution_108, unsqueeze_446, buf113, squeeze_235, buf111, primals_157, 392, 736, grid=grid(392, 736), stream=stream0)
        del buf97
        del clone_67
        del convolution_108
        del div_84
        del primals_157
        del squeeze_235
        del unsqueeze_446
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf117 = aten.convolution_backward(buf116, div_81, primals_315, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False])
        del div_81
        del primals_315
        buf118 = buf117[0]
        buf119 = buf117[1]
        del buf117
        buf120 = buf112; del buf112  # reuse
        buf122 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_30.run(clone_66, buf118, convolution_107, unsqueeze_458, buf120, buf122, 2944, 98, grid=grid(2944), stream=stream0)
        buf121 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_27.run(buf120, buf121, 736, 4, grid=grid(736), stream=stream0)
        buf123 = empty((736, ), device='cuda', dtype=torch.float32)
        buf124 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_28.run(buf122, squeeze_232, buf123, buf124, 736, 4, grid=grid(736), stream=stream0)
        buf125 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_31.run(clone_66, buf118, convolution_107, unsqueeze_458, buf123, squeeze_232, buf121, primals_155, buf125, 392, 736, grid=grid(392, 736), stream=stream0)
        del buf118
        del clone_66
        del convolution_107
        del primals_155
        del squeeze_232
        del unsqueeze_458
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf126 = aten.convolution_backward(buf125, add_486, primals_314, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_486
        del primals_314
        buf127 = buf126[0]
        buf128 = buf126[1]
        del buf126
        buf129 = buf93; del buf93  # reuse
        buf130 = empty((184, ), device='cuda', dtype=torch.float32)
        buf132 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_34.run(buf52, buf90, buf127, convolution_106, unsqueeze_470, squeeze_229, buf129, buf130, buf132, 184, 392, grid=grid(184), stream=stream0)
        buf131 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_35.run(buf52, buf90, buf127, convolution_106, unsqueeze_470, buf130, squeeze_229, buf129, primals_153, buf131, 1472, 49, grid=grid(1472, 49), stream=stream0)
        del convolution_106
        del primals_153
        del squeeze_229
        del unsqueeze_470
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf133 = aten.convolution_backward(buf131, mul_612, primals_313, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_612
        del primals_313
        buf134 = buf133[0]
        buf135 = buf133[1]
        del buf133
        buf136 = reinterpret_tensor(buf108, (8, 736, 1, 1), (736, 1, 5888, 5888), 0); del buf108  # reuse
        buf137 = reinterpret_tensor(buf136, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_24.run(buf137, buf134, div_78, bitwise_and_3, 5888, 49, grid=grid(5888), stream=stream0)
        del bitwise_and_3
        del div_78
        buf138 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_25.run(buf137, buf138, 736, 8, grid=grid(736), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf139 = aten.convolution_backward(buf137, div_79, primals_311, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf137
        del div_79
        del primals_311
        buf140 = buf139[0]
        buf141 = buf139[1]
        del buf139
        buf142 = buf140; del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_12.run(buf142, convolution_104, 384, grid=grid(384), stream=stream0)
        del convolution_104
        buf143 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_13.run(buf142, buf143, 48, 8, grid=grid(48), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf144 = aten.convolution_backward(buf142, mean_14, primals_309, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf142
        del mean_14
        del primals_309
        buf145 = buf144[0]
        buf146 = buf144[1]
        del buf144
        buf147 = buf122; del buf122  # reuse
        buf149 = buf120; del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_26.run(clone_64, buf134, div_80, buf145, convolution_103, unsqueeze_482, buf147, buf149, 2944, 98, grid=grid(2944), stream=stream0)
        buf148 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_27.run(buf147, buf148, 736, 4, grid=grid(736), stream=stream0)
        buf150 = empty((736, ), device='cuda', dtype=torch.float32)
        buf152 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_28.run(buf149, squeeze_226, buf150, buf152, 736, 4, grid=grid(736), stream=stream0)
        buf151 = buf125; del buf125  # reuse
        buf153 = buf151; del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_29.run(buf153, clone_64, buf134, div_80, buf145, convolution_103, unsqueeze_482, buf150, squeeze_226, buf148, primals_151, 392, 736, grid=grid(392, 736), stream=stream0)
        del buf134
        del clone_64
        del convolution_103
        del div_80
        del primals_151
        del squeeze_226
        del unsqueeze_482
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf154 = aten.convolution_backward(buf153, div_77, primals_308, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False])
        del div_77
        del primals_308
        buf155 = buf154[0]
        buf156 = buf154[1]
        del buf154
        buf157 = buf149; del buf149  # reuse
        buf159 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_30.run(clone_63, buf155, convolution_102, unsqueeze_494, buf157, buf159, 2944, 98, grid=grid(2944), stream=stream0)
        buf158 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_27.run(buf157, buf158, 736, 4, grid=grid(736), stream=stream0)
        buf160 = empty((736, ), device='cuda', dtype=torch.float32)
        buf161 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_28.run(buf159, squeeze_223, buf160, buf161, 736, 4, grid=grid(736), stream=stream0)
        buf162 = buf153; del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_31.run(clone_63, buf155, convolution_102, unsqueeze_494, buf160, squeeze_223, buf158, primals_149, buf162, 392, 736, grid=grid(392, 736), stream=stream0)
        del buf155
        del clone_63
        del convolution_102
        del primals_149
        del squeeze_223
        del unsqueeze_494
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf163 = aten.convolution_backward(buf162, add_466, primals_307, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_466
        del primals_307
        buf164 = buf163[0]
        buf165 = buf163[1]
        del buf163
        buf166 = buf130; del buf130  # reuse
        buf167 = empty((184, ), device='cuda', dtype=torch.float32)
        buf169 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_36.run(buf52, buf90, buf127, buf164, convolution_101, unsqueeze_506, squeeze_220, buf166, buf167, buf169, 184, 392, grid=grid(184), stream=stream0)
        buf168 = buf131; del buf131  # reuse
        buf170 = buf168; del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_37.run(buf170, buf52, buf90, buf127, buf164, convolution_101, unsqueeze_506, buf167, squeeze_220, buf166, primals_147, 1472, 49, grid=grid(1472, 49), stream=stream0)
        del convolution_101
        del primals_147
        del squeeze_220
        del unsqueeze_506
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf171 = aten.convolution_backward(buf170, mul_587, primals_306, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf170
        del mul_587
        del primals_306
        buf172 = buf171[0]
        buf173 = buf171[1]
        del buf171
        buf174 = reinterpret_tensor(buf145, (8, 736, 1, 1), (736, 1, 5888, 5888), 0); del buf145  # reuse
        buf175 = reinterpret_tensor(buf174, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_24.run(buf175, buf172, div_74, bitwise_and_4, 5888, 49, grid=grid(5888), stream=stream0)
        del bitwise_and_4
        del div_74
        buf176 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_25.run(buf175, buf176, 736, 8, grid=grid(736), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf177 = aten.convolution_backward(buf175, div_75, primals_304, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf175
        del div_75
        del primals_304
        buf178 = buf177[0]
        buf179 = buf177[1]
        del buf177
        buf180 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_12.run(buf180, convolution_99, 384, grid=grid(384), stream=stream0)
        del convolution_99
        buf181 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_13.run(buf180, buf181, 48, 8, grid=grid(48), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf182 = aten.convolution_backward(buf180, mean_13, primals_302, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf180
        del mean_13
        del primals_302
        buf183 = buf182[0]
        buf184 = buf182[1]
        del buf182
        buf185 = buf159; del buf159  # reuse
        buf187 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_26.run(clone_61, buf172, div_76, buf183, convolution_98, unsqueeze_518, buf185, buf187, 2944, 98, grid=grid(2944), stream=stream0)
        buf186 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_27.run(buf185, buf186, 736, 4, grid=grid(736), stream=stream0)
        buf188 = empty((736, ), device='cuda', dtype=torch.float32)
        buf190 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_28.run(buf187, squeeze_217, buf188, buf190, 736, 4, grid=grid(736), stream=stream0)
        buf189 = buf162; del buf162  # reuse
        buf191 = buf189; del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_29.run(buf191, clone_61, buf172, div_76, buf183, convolution_98, unsqueeze_518, buf188, squeeze_217, buf186, primals_145, 392, 736, grid=grid(392, 736), stream=stream0)
        del buf172
        del clone_61
        del convolution_98
        del div_76
        del primals_145
        del squeeze_217
        del unsqueeze_518
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf192 = aten.convolution_backward(buf191, div_73, primals_301, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False])
        del div_73
        del primals_301
        buf193 = buf192[0]
        buf194 = buf192[1]
        del buf192
        buf195 = buf187; del buf187  # reuse
        buf197 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_30.run(clone_60, buf193, convolution_97, unsqueeze_530, buf195, buf197, 2944, 98, grid=grid(2944), stream=stream0)
        buf196 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_27.run(buf195, buf196, 736, 4, grid=grid(736), stream=stream0)
        buf198 = empty((736, ), device='cuda', dtype=torch.float32)
        buf199 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_28.run(buf197, squeeze_214, buf198, buf199, 736, 4, grid=grid(736), stream=stream0)
        buf200 = buf191; del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_31.run(clone_60, buf193, convolution_97, unsqueeze_530, buf198, squeeze_214, buf196, primals_143, buf200, 392, 736, grid=grid(392, 736), stream=stream0)
        del buf193
        del clone_60
        del convolution_97
        del primals_143
        del squeeze_214
        del unsqueeze_530
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf201 = aten.convolution_backward(buf200, add_446, primals_300, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_446
        del primals_300
        buf202 = buf201[0]
        buf203 = buf201[1]
        del buf201
        buf204 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_38.run(buf204, buf52, buf90, buf164, buf202, 72128, grid=grid(72128), stream=stream0)
        del buf164
        del buf202
        del buf52
        buf205 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_20.run(buf204, buf205, 184, 392, grid=grid(184), stream=stream0)
        buf206 = reinterpret_tensor(buf198, (184, 4), (1, 184), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_21.run(buf204, convolution_96, unsqueeze_542, buf206, 736, 98, grid=grid(736), stream=stream0)
        buf207 = empty((184, ), device='cuda', dtype=torch.float32)
        buf208 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_22.run(buf206, squeeze_211, buf207, buf208, 184, 4, grid=grid(184), stream=stream0)
        buf209 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_23.run(buf204, convolution_96, unsqueeze_542, buf207, squeeze_211, buf205, primals_141, buf209, 1472, 49, grid=grid(1472, 49), stream=stream0)
        del convolution_96
        del primals_141
        del squeeze_211
        del unsqueeze_542
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf210 = aten.convolution_backward(buf209, mul_562, primals_299, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf209
        del mul_562
        del primals_299
        buf211 = buf210[0]
        buf212 = buf210[1]
        del buf210
        buf213 = reinterpret_tensor(buf183, (8, 736, 1, 1), (736, 1, 5888, 5888), 0); del buf183  # reuse
        buf214 = reinterpret_tensor(buf213, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_24.run(buf214, buf211, div_70, bitwise_and_5, 5888, 49, grid=grid(5888), stream=stream0)
        del bitwise_and_5
        del div_70
        buf215 = reinterpret_tensor(buf206, (736, ), (1, ), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_25.run(buf214, buf215, 736, 8, grid=grid(736), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf216 = aten.convolution_backward(buf214, div_71, primals_297, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf214
        del div_71
        del primals_297
        buf217 = buf216[0]
        buf218 = buf216[1]
        del buf216
        buf219 = buf217; del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_12.run(buf219, convolution_94, 384, grid=grid(384), stream=stream0)
        del convolution_94
        buf220 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_13.run(buf219, buf220, 48, 8, grid=grid(48), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf221 = aten.convolution_backward(buf219, mean_12, primals_295, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf219
        del mean_12
        del primals_295
        buf222 = buf221[0]
        buf223 = buf221[1]
        del buf221
        buf224 = buf197; del buf197  # reuse
        buf226 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_26.run(clone_58, buf211, div_72, buf222, convolution_93, unsqueeze_554, buf224, buf226, 2944, 98, grid=grid(2944), stream=stream0)
        buf225 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_27.run(buf224, buf225, 736, 4, grid=grid(736), stream=stream0)
        buf227 = empty((736, ), device='cuda', dtype=torch.float32)
        buf229 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_28.run(buf226, squeeze_208, buf227, buf229, 736, 4, grid=grid(736), stream=stream0)
        buf228 = buf200; del buf200  # reuse
        buf230 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_29.run(buf230, clone_58, buf211, div_72, buf222, convolution_93, unsqueeze_554, buf227, squeeze_208, buf225, primals_139, 392, 736, grid=grid(392, 736), stream=stream0)
        del buf211
        del buf222
        del clone_58
        del convolution_93
        del div_72
        del primals_139
        del squeeze_208
        del unsqueeze_554
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf231 = aten.convolution_backward(buf230, div_69, primals_294, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False])
        del div_69
        del primals_294
        buf232 = buf231[0]
        buf233 = buf231[1]
        del buf231
        buf234 = buf226; del buf226  # reuse
        buf236 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_30.run(clone_57, buf232, convolution_92, unsqueeze_566, buf234, buf236, 2944, 98, grid=grid(2944), stream=stream0)
        buf235 = buf227; del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_27.run(buf234, buf235, 736, 4, grid=grid(736), stream=stream0)
        del buf234
        buf237 = empty((736, ), device='cuda', dtype=torch.float32)
        buf238 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_28.run(buf236, squeeze_205, buf237, buf238, 736, 4, grid=grid(736), stream=stream0)
        del buf236
        buf239 = buf230; del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_31.run(clone_57, buf232, convolution_92, unsqueeze_566, buf237, squeeze_205, buf235, primals_137, buf239, 392, 736, grid=grid(392, 736), stream=stream0)
        del buf232
        del buf237
        del clone_57
        del convolution_92
        del primals_137
        del squeeze_205
        del unsqueeze_566
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf240 = aten.convolution_backward(buf239, add_426, primals_293, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_426
        del buf239
        del primals_293
        buf241 = buf240[0]
        buf242 = buf240[1]
        del buf240
        buf243 = buf207; del buf207  # reuse
        buf244 = empty((184, ), device='cuda', dtype=torch.float32)
        buf245 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_32.run(buf204, buf241, convolution_91, unsqueeze_578, squeeze_202, buf243, buf244, buf245, 184, 392, grid=grid(184), stream=stream0)
        buf246 = buf204; del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_39.run(buf246, buf241, convolution_91, unsqueeze_578, buf244, squeeze_202, buf243, primals_135, 1472, 49, grid=grid(1472, 49), stream=stream0)
        del buf241
        del buf244
        del convolution_91
        del primals_135
        del squeeze_202
        del unsqueeze_578
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf247 = aten.convolution_backward(buf246, mul_537, primals_292, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf246
        del mul_537
        del primals_292
        buf248 = buf247[0]
        buf249 = buf247[1]
        del buf247
        buf250 = empty_strided((8, 720, 1, 1), (720, 1, 5760, 5760), device='cuda', dtype=torch.float32)
        buf251 = reinterpret_tensor(buf250, (8, 720, 1, 1), (720, 1, 720, 720), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_40.run(buf251, buf248, div_66, bitwise_and_6, 5760, 49, grid=grid(5760), stream=stream0)
        del bitwise_and_6
        del div_66
        buf252 = empty((720, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_41.run(buf251, buf252, 720, 8, grid=grid(720), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf253 = aten.convolution_backward(buf251, div_67, primals_290, [720], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf251
        del div_67
        del primals_290
        buf254 = buf253[0]
        buf255 = buf253[1]
        del buf253
        buf256 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_42.run(buf256, convolution_89, 256, grid=grid(256), stream=stream0)
        del convolution_89
        buf257 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_43.run(buf256, buf257, 32, 8, grid=grid(32), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf258 = aten.convolution_backward(buf256, mean_11, primals_288, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf256
        del mean_11
        del primals_288
        buf259 = buf258[0]
        buf260 = buf258[1]
        del buf258
        buf261 = empty_strided((720, 4), (1, 720), device='cuda', dtype=torch.float32)
        buf263 = empty_strided((720, 4), (1, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_44.run(clone_55, buf248, div_68, buf259, convolution_88, unsqueeze_590, buf261, buf263, 2880, 98, grid=grid(2880), stream=stream0)
        buf262 = empty((720, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_45.run(buf261, buf262, 720, 4, grid=grid(720), stream=stream0)
        del buf261
        buf264 = empty((720, ), device='cuda', dtype=torch.float32)
        buf266 = empty((720, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_46.run(buf263, squeeze_199, buf264, buf266, 720, 4, grid=grid(720), stream=stream0)
        buf265 = empty_strided((8, 720, 7, 7), (35280, 1, 5040, 720), device='cuda', dtype=torch.float32)
        buf267 = buf265; del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_47.run(buf267, clone_55, buf248, div_68, buf259, convolution_88, unsqueeze_590, buf264, squeeze_199, buf262, primals_133, 392, 720, grid=grid(392, 720), stream=stream0)
        del buf248
        del clone_55
        del convolution_88
        del div_68
        del primals_133
        del squeeze_199
        del unsqueeze_590
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf268 = aten.convolution_backward(buf267, div_65, primals_287, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 720, [True, True, False])
        del buf267
        del div_65
        del primals_287
        buf269 = buf268[0]
        buf270 = buf268[1]
        del buf268
        buf271 = empty((720, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_48.run(clone_54, buf269, buf271, 9360, 121, grid=grid(9360), stream=stream0)
        buf272 = buf264; del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_49.run(buf271, buf272, 720, 13, grid=grid(720), stream=stream0)
        buf273 = reinterpret_tensor(buf271, (720, 13), (1, 720), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_50.run(clone_54, buf269, convolution_87, unsqueeze_602, buf273, 9360, 121, grid=grid(9360), stream=stream0)
        buf274 = empty((720, ), device='cuda', dtype=torch.float32)
        buf275 = empty((720, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_51.run(buf273, squeeze_196, buf274, buf275, 720, 13, grid=grid(720), stream=stream0)
        del buf273
        buf276 = empty_strided((8, 720, 14, 14), (141120, 1, 10080, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_52.run(clone_54, buf269, convolution_87, unsqueeze_602, buf274, squeeze_196, buf272, primals_131, buf276, 1568, 720, grid=grid(1568, 720), stream=stream0)
        del buf269
        del buf274
        del clone_54
        del convolution_87
        del primals_131
        del squeeze_196
        del unsqueeze_602
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf277 = aten.convolution_backward(buf276, add_407, primals_286, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_407
        del buf276
        del primals_286
        buf278 = buf277[0]
        buf279 = buf277[1]
        del buf277
        buf280 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf278, buf280, 120, 1568, grid=grid(120), stream=stream0)
        buf281 = empty((120, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_54.run(buf278, convolution_86, unsqueeze_614, buf281, 1560, 121, grid=grid(1560), stream=stream0)
        buf282 = empty((120, ), device='cuda', dtype=torch.float32)
        buf283 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_55.run(buf281, squeeze_193, buf282, buf283, 120, 13, grid=grid(120), stream=stream0)
        buf284 = empty((8, 120, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_56.run(buf278, convolution_86, unsqueeze_614, buf282, squeeze_193, buf280, primals_129, buf284, 960, 196, grid=grid(960, 196), stream=stream0)
        del convolution_86
        del primals_129
        del squeeze_193
        del unsqueeze_614
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf285 = aten.convolution_backward(buf284, mul_512, primals_285, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_512
        del primals_285
        buf286 = buf285[0]
        buf287 = buf285[1]
        del buf285
        buf288 = reinterpret_tensor(buf259, (8, 360, 1, 1, 2), (720, 2, 5760, 5760, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_57.run(buf286, div_62, buf288, 5760, 98, grid=grid(5760), stream=stream0)
        del div_62
        buf289 = reinterpret_tensor(buf263, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf263  # reuse
        buf290 = reinterpret_tensor(buf289, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_58.run(buf290, buf288, bitwise_and_7, 2880, 2, grid=grid(2880), stream=stream0)
        del bitwise_and_7
        buf291 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_59.run(buf290, buf291, 360, 8, grid=grid(360), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf292 = aten.convolution_backward(buf290, div_63, primals_283, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf290
        del div_63
        del primals_283
        buf293 = buf292[0]
        buf294 = buf292[1]
        del buf292
        buf295 = buf293; del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_42.run(buf295, convolution_84, 256, grid=grid(256), stream=stream0)
        del convolution_84
        buf296 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_43.run(buf295, buf296, 32, 8, grid=grid(32), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf297 = aten.convolution_backward(buf295, mean_10, primals_281, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf295
        del mean_10
        del primals_281
        buf298 = buf297[0]
        buf299 = buf297[1]
        del buf297
        buf300 = empty((360, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_60.run(clone_52, buf286, div_64, buf298, buf300, 4680, 121, grid=grid(4680), stream=stream0)
        buf301 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_61.run(buf300, buf301, 360, 13, grid=grid(360), stream=stream0)
        buf302 = reinterpret_tensor(buf300, (360, 13), (1, 360), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_62.run(clone_52, buf286, div_64, buf298, convolution_83, unsqueeze_626, buf302, 4680, 121, grid=grid(4680), stream=stream0)
        buf303 = empty((360, ), device='cuda', dtype=torch.float32)
        buf305 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_63.run(buf302, squeeze_190, buf303, buf305, 360, 13, grid=grid(360), stream=stream0)
        buf304 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        buf306 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_64.run(buf306, clone_52, buf286, div_64, buf298, convolution_83, unsqueeze_626, buf303, squeeze_190, buf301, primals_127, 1568, 360, grid=grid(1568, 360), stream=stream0)
        del buf286
        del clone_52
        del convolution_83
        del div_64
        del primals_127
        del squeeze_190
        del unsqueeze_626
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf307 = aten.convolution_backward(buf306, div_61, primals_280, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False])
        del div_61
        del primals_280
        buf308 = buf307[0]
        buf309 = buf307[1]
        del buf307
        buf310 = reinterpret_tensor(buf302, (360, 13), (13, 1), 0); del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_65.run(clone_51, buf308, buf310, 4680, 121, grid=grid(4680), stream=stream0)
        buf311 = buf303; del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_61.run(buf310, buf311, 360, 13, grid=grid(360), stream=stream0)
        buf312 = reinterpret_tensor(buf310, (360, 13), (1, 360), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_66.run(clone_51, buf308, convolution_82, unsqueeze_638, buf312, 4680, 121, grid=grid(4680), stream=stream0)
        buf313 = empty((360, ), device='cuda', dtype=torch.float32)
        buf314 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_63.run(buf312, squeeze_187, buf313, buf314, 360, 13, grid=grid(360), stream=stream0)
        buf315 = buf306; del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_67.run(clone_51, buf308, convolution_82, unsqueeze_638, buf313, squeeze_187, buf311, primals_125, buf315, 1568, 360, grid=grid(1568, 360), stream=stream0)
        del buf308
        del clone_51
        del convolution_82
        del primals_125
        del squeeze_187
        del unsqueeze_638
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf316 = aten.convolution_backward(buf315, add_387, primals_279, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_387
        del primals_279
        buf317 = buf316[0]
        buf318 = buf316[1]
        del buf316
        buf319 = buf282; del buf282  # reuse
        buf320 = empty((120, ), device='cuda', dtype=torch.float32)
        buf321 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_68.run(buf278, buf317, convolution_81, unsqueeze_650, squeeze_184, buf319, buf320, buf321, 120, 1568, grid=grid(120), stream=stream0)
        buf322 = buf284; del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_69.run(buf278, buf317, convolution_81, unsqueeze_650, buf320, squeeze_184, buf319, primals_123, buf322, 960, 196, grid=grid(960, 196), stream=stream0)
        del convolution_81
        del primals_123
        del squeeze_184
        del unsqueeze_650
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf323 = aten.convolution_backward(buf322, mul_487, primals_278, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_487
        del primals_278
        buf324 = buf323[0]
        buf325 = buf323[1]
        del buf323
        buf326 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_57.run(buf324, div_58, buf326, 5760, 98, grid=grid(5760), stream=stream0)
        del div_58
        buf327 = reinterpret_tensor(buf298, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf298  # reuse
        buf328 = reinterpret_tensor(buf327, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_58.run(buf328, buf326, bitwise_and_8, 2880, 2, grid=grid(2880), stream=stream0)
        del bitwise_and_8
        buf329 = buf313; del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_59.run(buf328, buf329, 360, 8, grid=grid(360), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf330 = aten.convolution_backward(buf328, div_59, primals_276, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf328
        del div_59
        del primals_276
        buf331 = buf330[0]
        buf332 = buf330[1]
        del buf330
        buf333 = buf331; del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_42.run(buf333, convolution_79, 256, grid=grid(256), stream=stream0)
        del convolution_79
        buf334 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_43.run(buf333, buf334, 32, 8, grid=grid(32), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf335 = aten.convolution_backward(buf333, mean_9, primals_274, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf333
        del mean_9
        del primals_274
        buf336 = buf335[0]
        buf337 = buf335[1]
        del buf335
        buf338 = reinterpret_tensor(buf312, (360, 13), (13, 1), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_60.run(clone_49, buf324, div_60, buf336, buf338, 4680, 121, grid=grid(4680), stream=stream0)
        buf339 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_61.run(buf338, buf339, 360, 13, grid=grid(360), stream=stream0)
        buf340 = reinterpret_tensor(buf338, (360, 13), (1, 360), 0); del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_62.run(clone_49, buf324, div_60, buf336, convolution_78, unsqueeze_662, buf340, 4680, 121, grid=grid(4680), stream=stream0)
        buf341 = empty((360, ), device='cuda', dtype=torch.float32)
        buf343 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_63.run(buf340, squeeze_181, buf341, buf343, 360, 13, grid=grid(360), stream=stream0)
        buf342 = buf315; del buf315  # reuse
        buf344 = buf342; del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_64.run(buf344, clone_49, buf324, div_60, buf336, convolution_78, unsqueeze_662, buf341, squeeze_181, buf339, primals_121, 1568, 360, grid=grid(1568, 360), stream=stream0)
        del buf324
        del clone_49
        del convolution_78
        del div_60
        del primals_121
        del squeeze_181
        del unsqueeze_662
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf345 = aten.convolution_backward(buf344, div_57, primals_273, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False])
        del div_57
        del primals_273
        buf346 = buf345[0]
        buf347 = buf345[1]
        del buf345
        buf348 = reinterpret_tensor(buf340, (360, 13), (13, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_65.run(clone_48, buf346, buf348, 4680, 121, grid=grid(4680), stream=stream0)
        buf349 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_61.run(buf348, buf349, 360, 13, grid=grid(360), stream=stream0)
        buf350 = reinterpret_tensor(buf348, (360, 13), (1, 360), 0); del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_66.run(clone_48, buf346, convolution_77, unsqueeze_674, buf350, 4680, 121, grid=grid(4680), stream=stream0)
        buf351 = empty((360, ), device='cuda', dtype=torch.float32)
        buf352 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_63.run(buf350, squeeze_178, buf351, buf352, 360, 13, grid=grid(360), stream=stream0)
        buf353 = buf344; del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_67.run(clone_48, buf346, convolution_77, unsqueeze_674, buf351, squeeze_178, buf349, primals_119, buf353, 1568, 360, grid=grid(1568, 360), stream=stream0)
        del buf346
        del clone_48
        del convolution_77
        del primals_119
        del squeeze_178
        del unsqueeze_674
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf354 = aten.convolution_backward(buf353, add_367, primals_272, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_367
        del primals_272
        buf355 = buf354[0]
        buf356 = buf354[1]
        del buf354
        buf357 = buf320; del buf320  # reuse
        buf358 = empty((120, ), device='cuda', dtype=torch.float32)
        buf360 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_70.run(buf278, buf317, buf355, convolution_76, unsqueeze_686, squeeze_175, buf357, buf358, buf360, 120, 1568, grid=grid(120), stream=stream0)
        buf359 = buf322; del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_71.run(buf278, buf317, buf355, convolution_76, unsqueeze_686, buf358, squeeze_175, buf357, primals_117, buf359, 960, 196, grid=grid(960, 196), stream=stream0)
        del convolution_76
        del primals_117
        del squeeze_175
        del unsqueeze_686
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf361 = aten.convolution_backward(buf359, mul_462, primals_271, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_462
        del primals_271
        buf362 = buf361[0]
        buf363 = buf361[1]
        del buf361
        buf364 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_57.run(buf362, div_54, buf364, 5760, 98, grid=grid(5760), stream=stream0)
        del div_54
        buf365 = reinterpret_tensor(buf336, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf336  # reuse
        buf366 = reinterpret_tensor(buf365, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_58.run(buf366, buf364, bitwise_and_9, 2880, 2, grid=grid(2880), stream=stream0)
        del bitwise_and_9
        buf367 = buf351; del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_59.run(buf366, buf367, 360, 8, grid=grid(360), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf368 = aten.convolution_backward(buf366, div_55, primals_269, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf366
        del div_55
        del primals_269
        buf369 = buf368[0]
        buf370 = buf368[1]
        del buf368
        buf371 = buf369; del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_42.run(buf371, convolution_74, 256, grid=grid(256), stream=stream0)
        del convolution_74
        buf372 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_43.run(buf371, buf372, 32, 8, grid=grid(32), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf373 = aten.convolution_backward(buf371, mean_8, primals_267, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf371
        del mean_8
        del primals_267
        buf374 = buf373[0]
        buf375 = buf373[1]
        del buf373
        buf376 = reinterpret_tensor(buf350, (360, 13), (13, 1), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_60.run(clone_46, buf362, div_56, buf374, buf376, 4680, 121, grid=grid(4680), stream=stream0)
        buf377 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_61.run(buf376, buf377, 360, 13, grid=grid(360), stream=stream0)
        buf378 = reinterpret_tensor(buf376, (360, 13), (1, 360), 0); del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_62.run(clone_46, buf362, div_56, buf374, convolution_73, unsqueeze_698, buf378, 4680, 121, grid=grid(4680), stream=stream0)
        buf379 = empty((360, ), device='cuda', dtype=torch.float32)
        buf381 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_63.run(buf378, squeeze_172, buf379, buf381, 360, 13, grid=grid(360), stream=stream0)
        buf380 = buf353; del buf353  # reuse
        buf382 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_64.run(buf382, clone_46, buf362, div_56, buf374, convolution_73, unsqueeze_698, buf379, squeeze_172, buf377, primals_115, 1568, 360, grid=grid(1568, 360), stream=stream0)
        del buf362
        del clone_46
        del convolution_73
        del div_56
        del primals_115
        del squeeze_172
        del unsqueeze_698
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf383 = aten.convolution_backward(buf382, div_53, primals_266, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False])
        del div_53
        del primals_266
        buf384 = buf383[0]
        buf385 = buf383[1]
        del buf383
        buf386 = reinterpret_tensor(buf378, (360, 13), (13, 1), 0); del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_65.run(clone_45, buf384, buf386, 4680, 121, grid=grid(4680), stream=stream0)
        buf387 = buf379; del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_61.run(buf386, buf387, 360, 13, grid=grid(360), stream=stream0)
        buf388 = reinterpret_tensor(buf386, (360, 13), (1, 360), 0); del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_66.run(clone_45, buf384, convolution_72, unsqueeze_710, buf388, 4680, 121, grid=grid(4680), stream=stream0)
        buf389 = empty((360, ), device='cuda', dtype=torch.float32)
        buf390 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_63.run(buf388, squeeze_169, buf389, buf390, 360, 13, grid=grid(360), stream=stream0)
        buf391 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_67.run(clone_45, buf384, convolution_72, unsqueeze_710, buf389, squeeze_169, buf387, primals_113, buf391, 1568, 360, grid=grid(1568, 360), stream=stream0)
        del buf384
        del clone_45
        del convolution_72
        del primals_113
        del squeeze_169
        del unsqueeze_710
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf392 = aten.convolution_backward(buf391, add_347, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_347
        del primals_265
        buf393 = buf392[0]
        buf394 = buf392[1]
        del buf392
        buf395 = buf358; del buf358  # reuse
        buf396 = empty((120, ), device='cuda', dtype=torch.float32)
        buf398 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_72.run(buf278, buf317, buf355, buf393, convolution_71, unsqueeze_722, squeeze_166, buf395, buf396, buf398, 120, 1568, grid=grid(120), stream=stream0)
        buf397 = buf359; del buf359  # reuse
        buf399 = buf397; del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_73.run(buf399, buf278, buf317, buf355, buf393, convolution_71, unsqueeze_722, buf396, squeeze_166, buf395, primals_111, 960, 196, grid=grid(960, 196), stream=stream0)
        del convolution_71
        del primals_111
        del squeeze_166
        del unsqueeze_722
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf400 = aten.convolution_backward(buf399, mul_437, primals_264, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf399
        del mul_437
        del primals_264
        buf401 = buf400[0]
        buf402 = buf400[1]
        del buf400
        buf403 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_57.run(buf401, div_50, buf403, 5760, 98, grid=grid(5760), stream=stream0)
        del div_50
        buf404 = reinterpret_tensor(buf374, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf374  # reuse
        buf405 = reinterpret_tensor(buf404, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_58.run(buf405, buf403, bitwise_and_10, 2880, 2, grid=grid(2880), stream=stream0)
        del bitwise_and_10
        buf406 = buf389; del buf389  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_59.run(buf405, buf406, 360, 8, grid=grid(360), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf407 = aten.convolution_backward(buf405, div_51, primals_262, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf405
        del div_51
        del primals_262
        buf408 = buf407[0]
        buf409 = buf407[1]
        del buf407
        buf410 = buf408; del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_42.run(buf410, convolution_69, 256, grid=grid(256), stream=stream0)
        del convolution_69
        buf411 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_43.run(buf410, buf411, 32, 8, grid=grid(32), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf412 = aten.convolution_backward(buf410, mean_7, primals_260, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf410
        del mean_7
        del primals_260
        buf413 = buf412[0]
        buf414 = buf412[1]
        del buf412
        buf415 = reinterpret_tensor(buf388, (360, 13), (13, 1), 0); del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_60.run(clone_43, buf401, div_52, buf413, buf415, 4680, 121, grid=grid(4680), stream=stream0)
        buf416 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_61.run(buf415, buf416, 360, 13, grid=grid(360), stream=stream0)
        buf417 = reinterpret_tensor(buf415, (360, 13), (1, 360), 0); del buf415  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_62.run(clone_43, buf401, div_52, buf413, convolution_68, unsqueeze_734, buf417, 4680, 121, grid=grid(4680), stream=stream0)
        buf418 = empty((360, ), device='cuda', dtype=torch.float32)
        buf420 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_63.run(buf417, squeeze_163, buf418, buf420, 360, 13, grid=grid(360), stream=stream0)
        buf419 = buf391; del buf391  # reuse
        buf421 = buf419; del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_64.run(buf421, clone_43, buf401, div_52, buf413, convolution_68, unsqueeze_734, buf418, squeeze_163, buf416, primals_109, 1568, 360, grid=grid(1568, 360), stream=stream0)
        del buf401
        del clone_43
        del convolution_68
        del div_52
        del primals_109
        del squeeze_163
        del unsqueeze_734
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf422 = aten.convolution_backward(buf421, div_49, primals_259, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False])
        del div_49
        del primals_259
        buf423 = buf422[0]
        buf424 = buf422[1]
        del buf422
        buf425 = reinterpret_tensor(buf417, (360, 13), (13, 1), 0); del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_65.run(clone_42, buf423, buf425, 4680, 121, grid=grid(4680), stream=stream0)
        buf426 = buf418; del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_61.run(buf425, buf426, 360, 13, grid=grid(360), stream=stream0)
        buf427 = reinterpret_tensor(buf425, (360, 13), (1, 360), 0); del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_66.run(clone_42, buf423, convolution_67, unsqueeze_746, buf427, 4680, 121, grid=grid(4680), stream=stream0)
        buf428 = empty((360, ), device='cuda', dtype=torch.float32)
        buf429 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_63.run(buf427, squeeze_160, buf428, buf429, 360, 13, grid=grid(360), stream=stream0)
        buf430 = buf421; del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_67.run(clone_42, buf423, convolution_67, unsqueeze_746, buf428, squeeze_160, buf426, primals_107, buf430, 1568, 360, grid=grid(1568, 360), stream=stream0)
        del buf423
        del clone_42
        del convolution_67
        del primals_107
        del squeeze_160
        del unsqueeze_746
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf431 = aten.convolution_backward(buf430, add_327, primals_258, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_327
        del primals_258
        buf432 = buf431[0]
        buf433 = buf431[1]
        del buf431
        buf434 = buf278; del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_74.run(buf434, buf317, buf355, buf393, buf432, 188160, grid=grid(188160), stream=stream0)
        del buf317
        del buf355
        del buf393
        buf435 = buf396; del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf434, buf435, 120, 1568, grid=grid(120), stream=stream0)
        buf436 = buf281; del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_54.run(buf434, convolution_66, unsqueeze_758, buf436, 1560, 121, grid=grid(1560), stream=stream0)
        buf437 = empty((120, ), device='cuda', dtype=torch.float32)
        buf438 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_55.run(buf436, squeeze_157, buf437, buf438, 120, 13, grid=grid(120), stream=stream0)
        del buf436
        buf439 = buf432; del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_56.run(buf434, convolution_66, unsqueeze_758, buf437, squeeze_157, buf435, primals_105, buf439, 960, 196, grid=grid(960, 196), stream=stream0)
        del convolution_66
        del primals_105
        del squeeze_157
        del unsqueeze_758
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf440 = aten.convolution_backward(buf439, mul_412, primals_257, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf439
        del mul_412
        del primals_257
        buf441 = buf440[0]
        buf442 = buf440[1]
        del buf440
        buf443 = buf403; del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_57.run(buf441, div_46, buf443, 5760, 98, grid=grid(5760), stream=stream0)
        del div_46
        buf444 = reinterpret_tensor(buf413, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf413  # reuse
        buf445 = reinterpret_tensor(buf444, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_58.run(buf445, buf443, bitwise_and_11, 2880, 2, grid=grid(2880), stream=stream0)
        del bitwise_and_11
        buf446 = buf428; del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_59.run(buf445, buf446, 360, 8, grid=grid(360), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf447 = aten.convolution_backward(buf445, div_47, primals_255, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf445
        del div_47
        del primals_255
        buf448 = buf447[0]
        buf449 = buf447[1]
        del buf447
        buf450 = buf448; del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_42.run(buf450, convolution_64, 256, grid=grid(256), stream=stream0)
        del convolution_64
        buf451 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_43.run(buf450, buf451, 32, 8, grid=grid(32), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf452 = aten.convolution_backward(buf450, mean_6, primals_253, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf450
        del mean_6
        del primals_253
        buf453 = buf452[0]
        buf454 = buf452[1]
        del buf452
        buf455 = reinterpret_tensor(buf427, (360, 13), (13, 1), 0); del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_60.run(clone_40, buf441, div_48, buf453, buf455, 4680, 121, grid=grid(4680), stream=stream0)
        buf456 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_61.run(buf455, buf456, 360, 13, grid=grid(360), stream=stream0)
        buf457 = reinterpret_tensor(buf455, (360, 13), (1, 360), 0); del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_62.run(clone_40, buf441, div_48, buf453, convolution_63, unsqueeze_770, buf457, 4680, 121, grid=grid(4680), stream=stream0)
        buf458 = empty((360, ), device='cuda', dtype=torch.float32)
        buf460 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_63.run(buf457, squeeze_154, buf458, buf460, 360, 13, grid=grid(360), stream=stream0)
        buf459 = buf430; del buf430  # reuse
        buf461 = buf459; del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_64.run(buf461, clone_40, buf441, div_48, buf453, convolution_63, unsqueeze_770, buf458, squeeze_154, buf456, primals_103, 1568, 360, grid=grid(1568, 360), stream=stream0)
        del buf441
        del clone_40
        del convolution_63
        del div_48
        del primals_103
        del squeeze_154
        del unsqueeze_770
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf462 = aten.convolution_backward(buf461, div_45, primals_252, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False])
        del div_45
        del primals_252
        buf463 = buf462[0]
        buf464 = buf462[1]
        del buf462
        buf465 = reinterpret_tensor(buf457, (360, 13), (13, 1), 0); del buf457  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_65.run(clone_39, buf463, buf465, 4680, 121, grid=grid(4680), stream=stream0)
        buf466 = buf458; del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_61.run(buf465, buf466, 360, 13, grid=grid(360), stream=stream0)
        buf467 = reinterpret_tensor(buf465, (360, 13), (1, 360), 0); del buf465  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_66.run(clone_39, buf463, convolution_62, unsqueeze_782, buf467, 4680, 121, grid=grid(4680), stream=stream0)
        buf468 = empty((360, ), device='cuda', dtype=torch.float32)
        buf469 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_63.run(buf467, squeeze_151, buf468, buf469, 360, 13, grid=grid(360), stream=stream0)
        buf470 = buf461; del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_67.run(clone_39, buf463, convolution_62, unsqueeze_782, buf468, squeeze_151, buf466, primals_101, buf470, 1568, 360, grid=grid(1568, 360), stream=stream0)
        del buf463
        del clone_39
        del convolution_62
        del primals_101
        del squeeze_151
        del unsqueeze_782
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf471 = aten.convolution_backward(buf470, add_307, primals_251, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_307
        del primals_251
        buf472 = buf471[0]
        buf473 = buf471[1]
        del buf471
        buf474 = buf437; del buf437  # reuse
        buf475 = empty((120, ), device='cuda', dtype=torch.float32)
        buf476 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_68.run(buf434, buf472, convolution_61, unsqueeze_794, squeeze_148, buf474, buf475, buf476, 120, 1568, grid=grid(120), stream=stream0)
        buf477 = buf434; del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_75.run(buf477, buf472, convolution_61, unsqueeze_794, buf475, squeeze_148, buf474, primals_99, 960, 196, grid=grid(960, 196), stream=stream0)
        del buf472
        del convolution_61
        del primals_99
        del squeeze_148
        del unsqueeze_794
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf478 = aten.convolution_backward(buf477, mul_387, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf477
        del mul_387
        del primals_250
        buf479 = buf478[0]
        buf480 = buf478[1]
        del buf478
        buf481 = buf443; del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_57.run(buf479, div_42, buf481, 5760, 98, grid=grid(5760), stream=stream0)
        del div_42
        buf482 = reinterpret_tensor(buf453, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf453  # reuse
        buf483 = reinterpret_tensor(buf482, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_58.run(buf483, buf481, bitwise_and_12, 2880, 2, grid=grid(2880), stream=stream0)
        del bitwise_and_12
        del buf481
        buf484 = buf468; del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_59.run(buf483, buf484, 360, 8, grid=grid(360), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf485 = aten.convolution_backward(buf483, div_43, primals_248, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf483
        del div_43
        del primals_248
        buf486 = buf485[0]
        buf487 = buf485[1]
        del buf485
        buf488 = buf486; del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_76.run(buf488, convolution_59, 192, grid=grid(192), stream=stream0)
        del convolution_59
        buf489 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_77.run(buf488, buf489, 24, 8, grid=grid(24), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf490 = aten.convolution_backward(buf488, mean_5, primals_246, [24], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf488
        del mean_5
        del primals_246
        buf491 = buf490[0]
        buf492 = buf490[1]
        del buf490
        buf493 = reinterpret_tensor(buf467, (360, 13), (13, 1), 0); del buf467  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_60.run(clone_37, buf479, div_44, buf491, buf493, 4680, 121, grid=grid(4680), stream=stream0)
        buf494 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_61.run(buf493, buf494, 360, 13, grid=grid(360), stream=stream0)
        buf495 = reinterpret_tensor(buf493, (360, 13), (1, 360), 0); del buf493  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_62.run(clone_37, buf479, div_44, buf491, convolution_58, unsqueeze_806, buf495, 4680, 121, grid=grid(4680), stream=stream0)
        buf496 = empty((360, ), device='cuda', dtype=torch.float32)
        buf498 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_63.run(buf495, squeeze_145, buf496, buf498, 360, 13, grid=grid(360), stream=stream0)
        buf497 = buf470; del buf470  # reuse
        buf499 = buf497; del buf497  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_64.run(buf499, clone_37, buf479, div_44, buf491, convolution_58, unsqueeze_806, buf496, squeeze_145, buf494, primals_97, 1568, 360, grid=grid(1568, 360), stream=stream0)
        del buf479
        del buf491
        del clone_37
        del convolution_58
        del div_44
        del primals_97
        del squeeze_145
        del unsqueeze_806
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf500 = aten.convolution_backward(buf499, div_41, primals_245, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 360, [True, True, False])
        del div_41
        del primals_245
        buf501 = buf500[0]
        buf502 = buf500[1]
        del buf500
        buf503 = reinterpret_tensor(buf495, (360, 13), (13, 1), 0); del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_65.run(clone_36, buf501, buf503, 4680, 121, grid=grid(4680), stream=stream0)
        buf504 = buf496; del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_61.run(buf503, buf504, 360, 13, grid=grid(360), stream=stream0)
        buf505 = reinterpret_tensor(buf503, (360, 13), (1, 360), 0); del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_66.run(clone_36, buf501, convolution_57, unsqueeze_818, buf505, 4680, 121, grid=grid(4680), stream=stream0)
        buf506 = empty((360, ), device='cuda', dtype=torch.float32)
        buf507 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_63.run(buf505, squeeze_142, buf506, buf507, 360, 13, grid=grid(360), stream=stream0)
        del buf505
        buf508 = buf499; del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_67.run(clone_36, buf501, convolution_57, unsqueeze_818, buf506, squeeze_142, buf504, primals_95, buf508, 1568, 360, grid=grid(1568, 360), stream=stream0)
        del buf501
        del buf506
        del clone_36
        del convolution_57
        del primals_95
        del squeeze_142
        del unsqueeze_818
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf509 = aten.convolution_backward(buf508, add_288, primals_244, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_288
        del buf508
        del primals_244
        buf510 = buf509[0]
        buf511 = buf509[1]
        del buf509
        buf512 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_78.run(buf510, buf512, 72, 1568, grid=grid(72), stream=stream0)
        buf513 = empty((72, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_79.run(buf510, convolution_56, unsqueeze_830, buf513, 936, 121, grid=grid(936), stream=stream0)
        buf514 = empty((72, ), device='cuda', dtype=torch.float32)
        buf515 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_80.run(buf513, squeeze_139, buf514, buf515, 72, 13, grid=grid(72), stream=stream0)
        buf516 = empty((8, 72, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_81.run(buf510, convolution_56, unsqueeze_830, buf514, squeeze_139, buf512, primals_93, buf516, 576, 196, grid=grid(576, 196), stream=stream0)
        del convolution_56
        del primals_93
        del squeeze_139
        del unsqueeze_830
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf517 = aten.convolution_backward(buf516, div_40, primals_243, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_40
        del primals_243
        buf518 = buf517[0]
        buf519 = buf517[1]
        del buf517
        buf520 = empty((216, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_82.run(clone_35, buf518, buf520, 2808, 121, grid=grid(2808), stream=stream0)
        buf521 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_83.run(buf520, buf521, 216, 13, grid=grid(216), stream=stream0)
        buf522 = reinterpret_tensor(buf520, (216, 13), (1, 216), 0); del buf520  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_84.run(clone_35, buf518, convolution_55, unsqueeze_842, buf522, 2808, 121, grid=grid(2808), stream=stream0)
        buf523 = empty((216, ), device='cuda', dtype=torch.float32)
        buf524 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_85.run(buf522, squeeze_136, buf523, buf524, 216, 13, grid=grid(216), stream=stream0)
        buf525 = empty_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_86.run(clone_35, buf518, convolution_55, unsqueeze_842, buf523, squeeze_136, buf521, primals_91, buf525, 1568, 216, grid=grid(1568, 216), stream=stream0)
        del buf518
        del clone_35
        del convolution_55
        del primals_91
        del squeeze_136
        del unsqueeze_842
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf526 = aten.convolution_backward(buf525, div_39, primals_242, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False])
        del div_39
        del primals_242
        buf527 = buf526[0]
        buf528 = buf526[1]
        del buf526
        buf529 = reinterpret_tensor(buf522, (216, 13), (13, 1), 0); del buf522  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_82.run(clone_34, buf527, buf529, 2808, 121, grid=grid(2808), stream=stream0)
        buf530 = buf523; del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_83.run(buf529, buf530, 216, 13, grid=grid(216), stream=stream0)
        buf531 = reinterpret_tensor(buf529, (216, 13), (1, 216), 0); del buf529  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_84.run(clone_34, buf527, convolution_54, unsqueeze_854, buf531, 2808, 121, grid=grid(2808), stream=stream0)
        buf532 = empty((216, ), device='cuda', dtype=torch.float32)
        buf533 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_85.run(buf531, squeeze_133, buf532, buf533, 216, 13, grid=grid(216), stream=stream0)
        buf534 = buf525; del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_86.run(clone_34, buf527, convolution_54, unsqueeze_854, buf532, squeeze_133, buf530, primals_89, buf534, 1568, 216, grid=grid(1568, 216), stream=stream0)
        del buf527
        del clone_34
        del convolution_54
        del primals_89
        del squeeze_133
        del unsqueeze_854
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf535 = aten.convolution_backward(buf534, add_270, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_270
        del primals_241
        buf536 = buf535[0]
        buf537 = buf535[1]
        del buf535
        buf538 = buf514; del buf514  # reuse
        buf539 = empty((72, ), device='cuda', dtype=torch.float32)
        buf540 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_87.run(buf510, buf536, convolution_53, unsqueeze_866, squeeze_130, buf538, buf539, buf540, 72, 1568, grid=grid(72), stream=stream0)
        buf541 = buf516; del buf516  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_88.run(buf510, buf536, convolution_53, unsqueeze_866, buf539, squeeze_130, buf538, primals_87, buf541, 576, 196, grid=grid(576, 196), stream=stream0)
        del convolution_53
        del primals_87
        del squeeze_130
        del unsqueeze_866
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf542 = aten.convolution_backward(buf541, div_38, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_38
        del primals_240
        buf543 = buf542[0]
        buf544 = buf542[1]
        del buf542
        buf545 = reinterpret_tensor(buf531, (216, 13), (13, 1), 0); del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_82.run(clone_33, buf543, buf545, 2808, 121, grid=grid(2808), stream=stream0)
        buf546 = buf532; del buf532  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_83.run(buf545, buf546, 216, 13, grid=grid(216), stream=stream0)
        buf547 = reinterpret_tensor(buf545, (216, 13), (1, 216), 0); del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_84.run(clone_33, buf543, convolution_52, unsqueeze_878, buf547, 2808, 121, grid=grid(2808), stream=stream0)
        buf548 = empty((216, ), device='cuda', dtype=torch.float32)
        buf549 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_85.run(buf547, squeeze_127, buf548, buf549, 216, 13, grid=grid(216), stream=stream0)
        buf550 = buf534; del buf534  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_86.run(clone_33, buf543, convolution_52, unsqueeze_878, buf548, squeeze_127, buf546, primals_85, buf550, 1568, 216, grid=grid(1568, 216), stream=stream0)
        del buf543
        del clone_33
        del convolution_52
        del primals_85
        del squeeze_127
        del unsqueeze_878
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf551 = aten.convolution_backward(buf550, div_37, primals_239, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False])
        del div_37
        del primals_239
        buf552 = buf551[0]
        buf553 = buf551[1]
        del buf551
        buf554 = reinterpret_tensor(buf547, (216, 13), (13, 1), 0); del buf547  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_82.run(clone_32, buf552, buf554, 2808, 121, grid=grid(2808), stream=stream0)
        buf555 = buf548; del buf548  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_83.run(buf554, buf555, 216, 13, grid=grid(216), stream=stream0)
        buf556 = reinterpret_tensor(buf554, (216, 13), (1, 216), 0); del buf554  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_84.run(clone_32, buf552, convolution_51, unsqueeze_890, buf556, 2808, 121, grid=grid(2808), stream=stream0)
        buf557 = empty((216, ), device='cuda', dtype=torch.float32)
        buf558 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_85.run(buf556, squeeze_124, buf557, buf558, 216, 13, grid=grid(216), stream=stream0)
        buf559 = buf550; del buf550  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_86.run(clone_32, buf552, convolution_51, unsqueeze_890, buf557, squeeze_124, buf555, primals_83, buf559, 1568, 216, grid=grid(1568, 216), stream=stream0)
        del buf552
        del clone_32
        del convolution_51
        del primals_83
        del squeeze_124
        del unsqueeze_890
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf560 = aten.convolution_backward(buf559, add_252, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_252
        del primals_238
        buf561 = buf560[0]
        buf562 = buf560[1]
        del buf560
        buf563 = buf539; del buf539  # reuse
        buf564 = empty((72, ), device='cuda', dtype=torch.float32)
        buf566 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_89.run(buf510, buf536, buf561, convolution_50, unsqueeze_902, squeeze_121, buf563, buf564, buf566, 72, 1568, grid=grid(72), stream=stream0)
        buf565 = buf541; del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_90.run(buf510, buf536, buf561, convolution_50, unsqueeze_902, buf564, squeeze_121, buf563, primals_81, buf565, 576, 196, grid=grid(576, 196), stream=stream0)
        del convolution_50
        del primals_81
        del squeeze_121
        del unsqueeze_902
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf567 = aten.convolution_backward(buf565, div_36, primals_237, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_36
        del primals_237
        buf568 = buf567[0]
        buf569 = buf567[1]
        del buf567
        buf570 = reinterpret_tensor(buf556, (216, 13), (13, 1), 0); del buf556  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_82.run(clone_31, buf568, buf570, 2808, 121, grid=grid(2808), stream=stream0)
        buf571 = buf557; del buf557  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_83.run(buf570, buf571, 216, 13, grid=grid(216), stream=stream0)
        buf572 = reinterpret_tensor(buf570, (216, 13), (1, 216), 0); del buf570  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_84.run(clone_31, buf568, convolution_49, unsqueeze_914, buf572, 2808, 121, grid=grid(2808), stream=stream0)
        buf573 = empty((216, ), device='cuda', dtype=torch.float32)
        buf574 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_85.run(buf572, squeeze_118, buf573, buf574, 216, 13, grid=grid(216), stream=stream0)
        buf575 = buf559; del buf559  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_86.run(clone_31, buf568, convolution_49, unsqueeze_914, buf573, squeeze_118, buf571, primals_79, buf575, 1568, 216, grid=grid(1568, 216), stream=stream0)
        del buf568
        del clone_31
        del convolution_49
        del primals_79
        del squeeze_118
        del unsqueeze_914
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf576 = aten.convolution_backward(buf575, div_35, primals_236, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False])
        del div_35
        del primals_236
        buf577 = buf576[0]
        buf578 = buf576[1]
        del buf576
        buf579 = reinterpret_tensor(buf572, (216, 13), (13, 1), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_82.run(clone_30, buf577, buf579, 2808, 121, grid=grid(2808), stream=stream0)
        buf580 = buf573; del buf573  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_83.run(buf579, buf580, 216, 13, grid=grid(216), stream=stream0)
        buf581 = reinterpret_tensor(buf579, (216, 13), (1, 216), 0); del buf579  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_84.run(clone_30, buf577, convolution_48, unsqueeze_926, buf581, 2808, 121, grid=grid(2808), stream=stream0)
        buf582 = empty((216, ), device='cuda', dtype=torch.float32)
        buf583 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_85.run(buf581, squeeze_115, buf582, buf583, 216, 13, grid=grid(216), stream=stream0)
        buf584 = buf575; del buf575  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_86.run(clone_30, buf577, convolution_48, unsqueeze_926, buf582, squeeze_115, buf580, primals_77, buf584, 1568, 216, grid=grid(1568, 216), stream=stream0)
        del buf577
        del clone_30
        del convolution_48
        del primals_77
        del squeeze_115
        del unsqueeze_926
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf585 = aten.convolution_backward(buf584, add_234, primals_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_234
        del primals_235
        buf586 = buf585[0]
        buf587 = buf585[1]
        del buf585
        buf588 = buf564; del buf564  # reuse
        buf589 = empty((72, ), device='cuda', dtype=torch.float32)
        buf591 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_91.run(buf510, buf536, buf561, buf586, convolution_47, unsqueeze_938, squeeze_112, buf588, buf589, buf591, 72, 1568, grid=grid(72), stream=stream0)
        buf590 = buf565; del buf565  # reuse
        buf592 = buf590; del buf590  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_92.run(buf592, buf510, buf536, buf561, buf586, convolution_47, unsqueeze_938, buf589, squeeze_112, buf588, primals_75, 576, 196, grid=grid(576, 196), stream=stream0)
        del convolution_47
        del primals_75
        del squeeze_112
        del unsqueeze_938
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf593 = aten.convolution_backward(buf592, div_34, primals_234, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf592
        del div_34
        del primals_234
        buf594 = buf593[0]
        buf595 = buf593[1]
        del buf593
        buf596 = reinterpret_tensor(buf581, (216, 13), (13, 1), 0); del buf581  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_82.run(clone_29, buf594, buf596, 2808, 121, grid=grid(2808), stream=stream0)
        buf597 = buf582; del buf582  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_83.run(buf596, buf597, 216, 13, grid=grid(216), stream=stream0)
        buf598 = reinterpret_tensor(buf596, (216, 13), (1, 216), 0); del buf596  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_84.run(clone_29, buf594, convolution_46, unsqueeze_950, buf598, 2808, 121, grid=grid(2808), stream=stream0)
        buf599 = empty((216, ), device='cuda', dtype=torch.float32)
        buf600 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_85.run(buf598, squeeze_109, buf599, buf600, 216, 13, grid=grid(216), stream=stream0)
        buf601 = buf584; del buf584  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_86.run(clone_29, buf594, convolution_46, unsqueeze_950, buf599, squeeze_109, buf597, primals_73, buf601, 1568, 216, grid=grid(1568, 216), stream=stream0)
        del buf594
        del clone_29
        del convolution_46
        del primals_73
        del squeeze_109
        del unsqueeze_950
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf602 = aten.convolution_backward(buf601, div_33, primals_233, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False])
        del div_33
        del primals_233
        buf603 = buf602[0]
        buf604 = buf602[1]
        del buf602
        buf605 = reinterpret_tensor(buf598, (216, 13), (13, 1), 0); del buf598  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_82.run(clone_28, buf603, buf605, 2808, 121, grid=grid(2808), stream=stream0)
        buf606 = buf599; del buf599  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_83.run(buf605, buf606, 216, 13, grid=grid(216), stream=stream0)
        buf607 = reinterpret_tensor(buf605, (216, 13), (1, 216), 0); del buf605  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_84.run(clone_28, buf603, convolution_45, unsqueeze_962, buf607, 2808, 121, grid=grid(2808), stream=stream0)
        buf608 = empty((216, ), device='cuda', dtype=torch.float32)
        buf609 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_85.run(buf607, squeeze_106, buf608, buf609, 216, 13, grid=grid(216), stream=stream0)
        del buf607
        buf610 = buf601; del buf601  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_86.run(clone_28, buf603, convolution_45, unsqueeze_962, buf608, squeeze_106, buf606, primals_71, buf610, 1568, 216, grid=grid(1568, 216), stream=stream0)
        del buf603
        del buf608
        del clone_28
        del convolution_45
        del primals_71
        del squeeze_106
        del unsqueeze_962
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf611 = aten.convolution_backward(buf610, add_216, primals_232, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_216
        del buf610
        del primals_232
        buf612 = buf611[0]
        buf613 = buf611[1]
        del buf611
        buf614 = buf510; del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_93.run(buf614, buf536, buf561, buf586, buf612, 112896, grid=grid(112896), stream=stream0)
        del buf536
        del buf561
        del buf586
        del buf612
        buf615 = buf589; del buf589  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_78.run(buf614, buf615, 72, 1568, grid=grid(72), stream=stream0)
        buf616 = buf513; del buf513  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_79.run(buf614, convolution_44, unsqueeze_974, buf616, 936, 121, grid=grid(936), stream=stream0)
        buf617 = empty((72, ), device='cuda', dtype=torch.float32)
        buf618 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_80.run(buf616, squeeze_103, buf617, buf618, 72, 13, grid=grid(72), stream=stream0)
        del buf616
        buf619 = buf614; del buf614  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_94.run(buf619, convolution_44, unsqueeze_974, buf617, squeeze_103, buf615, primals_69, 576, 196, grid=grid(576, 196), stream=stream0)
        del buf617
        del convolution_44
        del primals_69
        del squeeze_103
        del unsqueeze_974
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf620 = aten.convolution_backward(buf619, div_32, primals_231, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf619
        del div_32
        del primals_231
        buf621 = buf620[0]
        buf622 = buf620[1]
        del buf620
        buf623 = empty((200, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_95.run(clone_27, buf621, buf623, 2600, 121, grid=grid(2600), stream=stream0)
        buf624 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_96.run(buf623, buf624, 200, 13, grid=grid(200), stream=stream0)
        buf625 = reinterpret_tensor(buf623, (200, 13), (1, 200), 0); del buf623  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_97.run(clone_27, buf621, convolution_43, unsqueeze_986, buf625, 2600, 121, grid=grid(2600), stream=stream0)
        buf626 = empty((200, ), device='cuda', dtype=torch.float32)
        buf627 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_98.run(buf625, squeeze_100, buf626, buf627, 200, 13, grid=grid(200), stream=stream0)
        del buf625
        buf628 = empty_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_99.run(clone_27, buf621, convolution_43, unsqueeze_986, buf626, squeeze_100, buf624, primals_67, buf628, 1568, 200, grid=grid(1568, 200), stream=stream0)
        del buf621
        del clone_27
        del convolution_43
        del primals_67
        del squeeze_100
        del unsqueeze_986
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf629 = aten.convolution_backward(buf628, div_31, primals_230, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 200, [True, True, False])
        del buf628
        del div_31
        del primals_230
        buf630 = buf629[0]
        buf631 = buf629[1]
        del buf629
        buf632 = empty((200, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_100.run(clone_26, buf630, buf632, 9800, 128, grid=grid(9800), stream=stream0)
        buf633 = buf626; del buf626  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_101.run(buf632, buf633, 200, 49, grid=grid(200), stream=stream0)
        buf634 = reinterpret_tensor(buf632, (200, 49), (1, 200), 0); del buf632  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_102.run(clone_26, buf630, convolution_42, unsqueeze_998, buf634, 9800, 128, grid=grid(9800), stream=stream0)
        buf635 = empty((200, ), device='cuda', dtype=torch.float32)
        buf636 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_103.run(buf634, squeeze_97, buf635, buf636, 200, 49, grid=grid(200), stream=stream0)
        del buf634
        buf637 = empty_strided((8, 200, 28, 28), (156800, 1, 5600, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_104.run(clone_26, buf630, convolution_42, unsqueeze_998, buf635, squeeze_97, buf633, primals_65, buf637, 6272, 200, grid=grid(6272, 200), stream=stream0)
        del buf630
        del buf635
        del clone_26
        del convolution_42
        del primals_65
        del squeeze_97
        del unsqueeze_998
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf638 = aten.convolution_backward(buf637, add_199, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_199
        del buf637
        del primals_229
        buf639 = buf638[0]
        buf640 = buf638[1]
        del buf638
        buf641 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_105.run(buf639, buf641, 40, 6272, grid=grid(40), stream=stream0)
        buf642 = empty((40, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_106.run(buf639, convolution_41, unsqueeze_1010, buf642, 1960, 128, grid=grid(1960), stream=stream0)
        buf643 = empty((40, ), device='cuda', dtype=torch.float32)
        buf644 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_107.run(buf642, squeeze_94, buf643, buf644, 40, 49, grid=grid(40), stream=stream0)
        buf645 = empty((8, 40, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_108.run(buf639, convolution_41, unsqueeze_1010, buf643, squeeze_94, buf641, primals_63, buf645, 320, 784, grid=grid(320, 784), stream=stream0)
        del convolution_41
        del primals_63
        del squeeze_94
        del unsqueeze_1010
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf646 = aten.convolution_backward(buf645, mul_247, primals_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_247
        del primals_228
        buf647 = buf646[0]
        buf648 = buf646[1]
        del buf646
        buf649 = empty_strided((8, 120, 1, 1, 7), (840, 7, 6720, 6720, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_109.run(buf647, div_28, buf649, 6720, 112, grid=grid(6720), stream=stream0)
        del div_28
        buf650 = empty_strided((8, 120, 1, 1), (120, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf651 = reinterpret_tensor(buf650, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf650  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_110.run(buf651, buf649, bitwise_and_13, 960, 7, grid=grid(960), stream=stream0)
        del bitwise_and_13
        buf652 = buf475; del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_111.run(buf651, buf652, 120, 8, grid=grid(120), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf653 = aten.convolution_backward(buf651, div_29, primals_226, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf651
        del div_29
        del primals_226
        buf654 = buf653[0]
        buf655 = buf653[1]
        del buf653
        buf656 = buf654; del buf654  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_112.run(buf656, convolution_39, 128, grid=grid(128), stream=stream0)
        del convolution_39
        buf657 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_113.run(buf656, buf657, 16, 8, grid=grid(16), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf658 = aten.convolution_backward(buf656, mean_4, primals_224, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf656
        del mean_4
        del primals_224
        buf659 = buf658[0]
        buf660 = buf658[1]
        del buf658
        buf661 = empty((120, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_114.run(clone_24, buf647, div_30, buf659, buf661, 5880, 128, grid=grid(5880), stream=stream0)
        buf662 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_115.run(buf661, buf662, 120, 49, grid=grid(120), stream=stream0)
        buf663 = reinterpret_tensor(buf661, (120, 49), (1, 120), 0); del buf661  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_116.run(clone_24, buf647, div_30, buf659, convolution_38, unsqueeze_1022, buf663, 5880, 128, grid=grid(5880), stream=stream0)
        buf664 = empty((120, ), device='cuda', dtype=torch.float32)
        buf666 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_117.run(buf663, squeeze_91, buf664, buf666, 120, 49, grid=grid(120), stream=stream0)
        buf665 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        buf667 = buf665; del buf665  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_118.run(buf667, clone_24, buf647, div_30, buf659, convolution_38, unsqueeze_1022, buf664, squeeze_91, buf662, primals_61, 6272, 120, grid=grid(6272, 120), stream=stream0)
        del buf647
        del clone_24
        del convolution_38
        del div_30
        del primals_61
        del squeeze_91
        del unsqueeze_1022
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf668 = aten.convolution_backward(buf667, div_27, primals_223, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del div_27
        del primals_223
        buf669 = buf668[0]
        buf670 = buf668[1]
        del buf668
        buf671 = reinterpret_tensor(buf663, (120, 49), (49, 1), 0); del buf663  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_119.run(clone_23, buf669, buf671, 5880, 128, grid=grid(5880), stream=stream0)
        buf672 = buf664; del buf664  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_115.run(buf671, buf672, 120, 49, grid=grid(120), stream=stream0)
        buf673 = reinterpret_tensor(buf671, (120, 49), (1, 120), 0); del buf671  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_120.run(clone_23, buf669, convolution_37, unsqueeze_1034, buf673, 5880, 128, grid=grid(5880), stream=stream0)
        buf674 = empty((120, ), device='cuda', dtype=torch.float32)
        buf675 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_117.run(buf673, squeeze_88, buf674, buf675, 120, 49, grid=grid(120), stream=stream0)
        buf676 = buf667; del buf667  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_121.run(clone_23, buf669, convolution_37, unsqueeze_1034, buf674, squeeze_88, buf672, primals_59, buf676, 6272, 120, grid=grid(6272, 120), stream=stream0)
        del buf669
        del clone_23
        del convolution_37
        del primals_59
        del squeeze_88
        del unsqueeze_1034
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf677 = aten.convolution_backward(buf676, add_179, primals_222, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_179
        del primals_222
        buf678 = buf677[0]
        buf679 = buf677[1]
        del buf677
        buf680 = buf643; del buf643  # reuse
        buf681 = empty((40, ), device='cuda', dtype=torch.float32)
        buf682 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_122.run(buf639, buf678, convolution_36, unsqueeze_1046, squeeze_85, buf680, buf681, buf682, 40, 6272, grid=grid(40), stream=stream0)
        buf683 = buf645; del buf645  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_123.run(buf639, buf678, convolution_36, unsqueeze_1046, buf681, squeeze_85, buf680, primals_57, buf683, 320, 784, grid=grid(320, 784), stream=stream0)
        del convolution_36
        del primals_57
        del squeeze_85
        del unsqueeze_1046
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf684 = aten.convolution_backward(buf683, mul_222, primals_221, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_222
        del primals_221
        buf685 = buf684[0]
        buf686 = buf684[1]
        del buf684
        buf687 = buf649; del buf649  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_109.run(buf685, div_24, buf687, 6720, 112, grid=grid(6720), stream=stream0)
        del div_24
        buf688 = reinterpret_tensor(buf659, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf659  # reuse
        buf689 = reinterpret_tensor(buf688, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf688  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_110.run(buf689, buf687, bitwise_and_14, 960, 7, grid=grid(960), stream=stream0)
        del bitwise_and_14
        buf690 = buf674; del buf674  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_111.run(buf689, buf690, 120, 8, grid=grid(120), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf691 = aten.convolution_backward(buf689, div_25, primals_219, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf689
        del div_25
        del primals_219
        buf692 = buf691[0]
        buf693 = buf691[1]
        del buf691
        buf694 = buf692; del buf692  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_112.run(buf694, convolution_34, 128, grid=grid(128), stream=stream0)
        del convolution_34
        buf695 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_113.run(buf694, buf695, 16, 8, grid=grid(16), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf696 = aten.convolution_backward(buf694, mean_3, primals_217, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf694
        del mean_3
        del primals_217
        buf697 = buf696[0]
        buf698 = buf696[1]
        del buf696
        buf699 = reinterpret_tensor(buf673, (120, 49), (49, 1), 0); del buf673  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_114.run(clone_21, buf685, div_26, buf697, buf699, 5880, 128, grid=grid(5880), stream=stream0)
        buf700 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_115.run(buf699, buf700, 120, 49, grid=grid(120), stream=stream0)
        buf701 = reinterpret_tensor(buf699, (120, 49), (1, 120), 0); del buf699  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_116.run(clone_21, buf685, div_26, buf697, convolution_33, unsqueeze_1058, buf701, 5880, 128, grid=grid(5880), stream=stream0)
        buf702 = empty((120, ), device='cuda', dtype=torch.float32)
        buf704 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_117.run(buf701, squeeze_82, buf702, buf704, 120, 49, grid=grid(120), stream=stream0)
        buf703 = buf676; del buf676  # reuse
        buf705 = buf703; del buf703  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_118.run(buf705, clone_21, buf685, div_26, buf697, convolution_33, unsqueeze_1058, buf702, squeeze_82, buf700, primals_55, 6272, 120, grid=grid(6272, 120), stream=stream0)
        del buf685
        del clone_21
        del convolution_33
        del div_26
        del primals_55
        del squeeze_82
        del unsqueeze_1058
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf706 = aten.convolution_backward(buf705, div_23, primals_216, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del div_23
        del primals_216
        buf707 = buf706[0]
        buf708 = buf706[1]
        del buf706
        buf709 = reinterpret_tensor(buf701, (120, 49), (49, 1), 0); del buf701  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_119.run(clone_20, buf707, buf709, 5880, 128, grid=grid(5880), stream=stream0)
        buf710 = buf702; del buf702  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_115.run(buf709, buf710, 120, 49, grid=grid(120), stream=stream0)
        buf711 = reinterpret_tensor(buf709, (120, 49), (1, 120), 0); del buf709  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_120.run(clone_20, buf707, convolution_32, unsqueeze_1070, buf711, 5880, 128, grid=grid(5880), stream=stream0)
        buf712 = empty((120, ), device='cuda', dtype=torch.float32)
        buf713 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_117.run(buf711, squeeze_79, buf712, buf713, 120, 49, grid=grid(120), stream=stream0)
        buf714 = buf705; del buf705  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_121.run(clone_20, buf707, convolution_32, unsqueeze_1070, buf712, squeeze_79, buf710, primals_53, buf714, 6272, 120, grid=grid(6272, 120), stream=stream0)
        del buf707
        del clone_20
        del convolution_32
        del primals_53
        del squeeze_79
        del unsqueeze_1070
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf715 = aten.convolution_backward(buf714, add_159, primals_215, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_159
        del primals_215
        buf716 = buf715[0]
        buf717 = buf715[1]
        del buf715
        buf718 = buf681; del buf681  # reuse
        buf719 = empty((40, ), device='cuda', dtype=torch.float32)
        buf721 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_124.run(buf639, buf678, buf716, convolution_31, unsqueeze_1082, squeeze_76, buf718, buf719, buf721, 40, 6272, grid=grid(40), stream=stream0)
        buf720 = buf683; del buf683  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_125.run(buf639, buf678, buf716, convolution_31, unsqueeze_1082, buf719, squeeze_76, buf718, primals_51, buf720, 320, 784, grid=grid(320, 784), stream=stream0)
        del convolution_31
        del primals_51
        del squeeze_76
        del unsqueeze_1082
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf722 = aten.convolution_backward(buf720, mul_197, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_197
        del primals_214
        buf723 = buf722[0]
        buf724 = buf722[1]
        del buf722
        buf725 = buf687; del buf687  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_109.run(buf723, div_20, buf725, 6720, 112, grid=grid(6720), stream=stream0)
        del div_20
        buf726 = reinterpret_tensor(buf697, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf697  # reuse
        buf727 = reinterpret_tensor(buf726, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf726  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_110.run(buf727, buf725, bitwise_and_15, 960, 7, grid=grid(960), stream=stream0)
        del bitwise_and_15
        buf728 = buf712; del buf712  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_111.run(buf727, buf728, 120, 8, grid=grid(120), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf729 = aten.convolution_backward(buf727, div_21, primals_212, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf727
        del div_21
        del primals_212
        buf730 = buf729[0]
        buf731 = buf729[1]
        del buf729
        buf732 = buf730; del buf730  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_112.run(buf732, convolution_29, 128, grid=grid(128), stream=stream0)
        del convolution_29
        buf733 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_113.run(buf732, buf733, 16, 8, grid=grid(16), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf734 = aten.convolution_backward(buf732, mean_2, primals_210, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf732
        del mean_2
        del primals_210
        buf735 = buf734[0]
        buf736 = buf734[1]
        del buf734
        buf737 = reinterpret_tensor(buf711, (120, 49), (49, 1), 0); del buf711  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_114.run(clone_18, buf723, div_22, buf735, buf737, 5880, 128, grid=grid(5880), stream=stream0)
        buf738 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_115.run(buf737, buf738, 120, 49, grid=grid(120), stream=stream0)
        buf739 = reinterpret_tensor(buf737, (120, 49), (1, 120), 0); del buf737  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_116.run(clone_18, buf723, div_22, buf735, convolution_28, unsqueeze_1094, buf739, 5880, 128, grid=grid(5880), stream=stream0)
        buf740 = empty((120, ), device='cuda', dtype=torch.float32)
        buf742 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_117.run(buf739, squeeze_73, buf740, buf742, 120, 49, grid=grid(120), stream=stream0)
        buf741 = buf714; del buf714  # reuse
        buf743 = buf741; del buf741  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_118.run(buf743, clone_18, buf723, div_22, buf735, convolution_28, unsqueeze_1094, buf740, squeeze_73, buf738, primals_49, 6272, 120, grid=grid(6272, 120), stream=stream0)
        del buf723
        del clone_18
        del convolution_28
        del div_22
        del primals_49
        del squeeze_73
        del unsqueeze_1094
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf744 = aten.convolution_backward(buf743, div_19, primals_209, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del div_19
        del primals_209
        buf745 = buf744[0]
        buf746 = buf744[1]
        del buf744
        buf747 = reinterpret_tensor(buf739, (120, 49), (49, 1), 0); del buf739  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_119.run(clone_17, buf745, buf747, 5880, 128, grid=grid(5880), stream=stream0)
        buf748 = buf740; del buf740  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_115.run(buf747, buf748, 120, 49, grid=grid(120), stream=stream0)
        buf749 = reinterpret_tensor(buf747, (120, 49), (1, 120), 0); del buf747  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_120.run(clone_17, buf745, convolution_27, unsqueeze_1106, buf749, 5880, 128, grid=grid(5880), stream=stream0)
        buf750 = empty((120, ), device='cuda', dtype=torch.float32)
        buf751 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_117.run(buf749, squeeze_70, buf750, buf751, 120, 49, grid=grid(120), stream=stream0)
        buf752 = buf743; del buf743  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_121.run(clone_17, buf745, convolution_27, unsqueeze_1106, buf750, squeeze_70, buf748, primals_47, buf752, 6272, 120, grid=grid(6272, 120), stream=stream0)
        del buf745
        del clone_17
        del convolution_27
        del primals_47
        del squeeze_70
        del unsqueeze_1106
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf753 = aten.convolution_backward(buf752, add_139, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_139
        del primals_208
        buf754 = buf753[0]
        buf755 = buf753[1]
        del buf753
        buf756 = buf719; del buf719  # reuse
        buf757 = empty((40, ), device='cuda', dtype=torch.float32)
        buf759 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_126.run(buf639, buf678, buf716, buf754, convolution_26, unsqueeze_1118, squeeze_67, buf756, buf757, buf759, 40, 6272, grid=grid(40), stream=stream0)
        buf758 = buf720; del buf720  # reuse
        buf760 = buf758; del buf758  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_127.run(buf760, buf639, buf678, buf716, buf754, convolution_26, unsqueeze_1118, buf757, squeeze_67, buf756, primals_45, 320, 784, grid=grid(320, 784), stream=stream0)
        del convolution_26
        del primals_45
        del squeeze_67
        del unsqueeze_1118
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf761 = aten.convolution_backward(buf760, mul_172, primals_207, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf760
        del mul_172
        del primals_207
        buf762 = buf761[0]
        buf763 = buf761[1]
        del buf761
        buf764 = buf725; del buf725  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_109.run(buf762, div_16, buf764, 6720, 112, grid=grid(6720), stream=stream0)
        del div_16
        buf765 = reinterpret_tensor(buf735, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf735  # reuse
        buf766 = reinterpret_tensor(buf765, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf765  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_110.run(buf766, buf764, bitwise_and_16, 960, 7, grid=grid(960), stream=stream0)
        del bitwise_and_16
        buf767 = buf750; del buf750  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_111.run(buf766, buf767, 120, 8, grid=grid(120), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf768 = aten.convolution_backward(buf766, div_17, primals_205, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf766
        del div_17
        del primals_205
        buf769 = buf768[0]
        buf770 = buf768[1]
        del buf768
        buf771 = buf769; del buf769  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_112.run(buf771, convolution_24, 128, grid=grid(128), stream=stream0)
        del convolution_24
        buf772 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_113.run(buf771, buf772, 16, 8, grid=grid(16), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf773 = aten.convolution_backward(buf771, mean_1, primals_203, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf771
        del mean_1
        del primals_203
        buf774 = buf773[0]
        buf775 = buf773[1]
        del buf773
        buf776 = reinterpret_tensor(buf749, (120, 49), (49, 1), 0); del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_114.run(clone_15, buf762, div_18, buf774, buf776, 5880, 128, grid=grid(5880), stream=stream0)
        buf777 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_115.run(buf776, buf777, 120, 49, grid=grid(120), stream=stream0)
        buf778 = reinterpret_tensor(buf776, (120, 49), (1, 120), 0); del buf776  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_116.run(clone_15, buf762, div_18, buf774, convolution_23, unsqueeze_1130, buf778, 5880, 128, grid=grid(5880), stream=stream0)
        buf779 = empty((120, ), device='cuda', dtype=torch.float32)
        buf781 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_117.run(buf778, squeeze_64, buf779, buf781, 120, 49, grid=grid(120), stream=stream0)
        buf780 = buf752; del buf752  # reuse
        buf782 = buf780; del buf780  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_118.run(buf782, clone_15, buf762, div_18, buf774, convolution_23, unsqueeze_1130, buf779, squeeze_64, buf777, primals_43, 6272, 120, grid=grid(6272, 120), stream=stream0)
        del buf762
        del clone_15
        del convolution_23
        del div_18
        del primals_43
        del squeeze_64
        del unsqueeze_1130
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf783 = aten.convolution_backward(buf782, div_15, primals_202, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del div_15
        del primals_202
        buf784 = buf783[0]
        buf785 = buf783[1]
        del buf783
        buf786 = reinterpret_tensor(buf778, (120, 49), (49, 1), 0); del buf778  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_119.run(clone_14, buf784, buf786, 5880, 128, grid=grid(5880), stream=stream0)
        buf787 = buf779; del buf779  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_115.run(buf786, buf787, 120, 49, grid=grid(120), stream=stream0)
        buf788 = reinterpret_tensor(buf786, (120, 49), (1, 120), 0); del buf786  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_120.run(clone_14, buf784, convolution_22, unsqueeze_1142, buf788, 5880, 128, grid=grid(5880), stream=stream0)
        buf789 = empty((120, ), device='cuda', dtype=torch.float32)
        buf790 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_117.run(buf788, squeeze_61, buf789, buf790, 120, 49, grid=grid(120), stream=stream0)
        buf791 = buf782; del buf782  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_121.run(clone_14, buf784, convolution_22, unsqueeze_1142, buf789, squeeze_61, buf787, primals_41, buf791, 6272, 120, grid=grid(6272, 120), stream=stream0)
        del buf784
        del clone_14
        del convolution_22
        del primals_41
        del squeeze_61
        del unsqueeze_1142
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf792 = aten.convolution_backward(buf791, add_119, primals_201, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_119
        del primals_201
        buf793 = buf792[0]
        buf794 = buf792[1]
        del buf792
        buf795 = buf639; del buf639  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_128.run(buf795, buf678, buf716, buf754, buf793, 250880, grid=grid(250880), stream=stream0)
        del buf678
        del buf716
        del buf754
        del buf793
        buf796 = buf757; del buf757  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_105.run(buf795, buf796, 40, 6272, grid=grid(40), stream=stream0)
        buf797 = buf642; del buf642  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_106.run(buf795, convolution_21, unsqueeze_1154, buf797, 1960, 128, grid=grid(1960), stream=stream0)
        buf798 = empty((40, ), device='cuda', dtype=torch.float32)
        buf799 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_107.run(buf797, squeeze_58, buf798, buf799, 40, 49, grid=grid(40), stream=stream0)
        del buf797
        buf800 = buf795; del buf795  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_129.run(buf800, convolution_21, unsqueeze_1154, buf798, squeeze_58, buf796, primals_39, 320, 784, grid=grid(320, 784), stream=stream0)
        del buf798
        del convolution_21
        del primals_39
        del squeeze_58
        del unsqueeze_1154
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf801 = aten.convolution_backward(buf800, mul_147, primals_200, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf800
        del mul_147
        del primals_200
        buf802 = buf801[0]
        buf803 = buf801[1]
        del buf801
        buf804 = buf764; del buf764  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_109.run(buf802, div_12, buf804, 6720, 112, grid=grid(6720), stream=stream0)
        del div_12
        buf805 = reinterpret_tensor(buf774, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf774  # reuse
        buf806 = reinterpret_tensor(buf805, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf805  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_110.run(buf806, buf804, bitwise_and_17, 960, 7, grid=grid(960), stream=stream0)
        del bitwise_and_17
        del buf804
        buf807 = buf789; del buf789  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_111.run(buf806, buf807, 120, 8, grid=grid(120), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf808 = aten.convolution_backward(buf806, div_13, primals_198, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf806
        del div_13
        del primals_198
        buf809 = buf808[0]
        buf810 = buf808[1]
        del buf808
        buf811 = buf809; del buf809  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_130.run(buf811, convolution_19, 64, grid=grid(64), stream=stream0)
        del convolution_19
        buf812 = empty((8, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_131.run(buf811, buf812, 8, 8, grid=grid(8), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf813 = aten.convolution_backward(buf811, mean, primals_196, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean
        del primals_196
        buf814 = buf813[0]
        buf815 = buf813[1]
        del buf813
        buf816 = reinterpret_tensor(buf788, (120, 49), (49, 1), 0); del buf788  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_114.run(clone_12, buf802, div_14, buf814, buf816, 5880, 128, grid=grid(5880), stream=stream0)
        buf817 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_115.run(buf816, buf817, 120, 49, grid=grid(120), stream=stream0)
        buf818 = reinterpret_tensor(buf816, (120, 49), (1, 120), 0); del buf816  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_116.run(clone_12, buf802, div_14, buf814, convolution_18, unsqueeze_1166, buf818, 5880, 128, grid=grid(5880), stream=stream0)
        buf819 = empty((120, ), device='cuda', dtype=torch.float32)
        buf821 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_117.run(buf818, squeeze_55, buf819, buf821, 120, 49, grid=grid(120), stream=stream0)
        del buf818
        buf820 = buf791; del buf791  # reuse
        buf822 = buf820; del buf820  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_118.run(buf822, clone_12, buf802, div_14, buf814, convolution_18, unsqueeze_1166, buf819, squeeze_55, buf817, primals_37, 6272, 120, grid=grid(6272, 120), stream=stream0)
        del buf802
        del buf814
        del clone_12
        del convolution_18
        del div_14
        del primals_37
        del squeeze_55
        del unsqueeze_1166
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf823 = aten.convolution_backward(buf822, div_11, primals_195, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del buf822
        del div_11
        del primals_195
        buf824 = buf823[0]
        buf825 = buf823[1]
        del buf823
        buf826 = empty((120, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_132.run(clone_11, buf824, buf826, 23520, 128, grid=grid(23520), stream=stream0)
        buf827 = buf819; del buf819  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_133.run(buf826, buf827, 120, 196, grid=grid(120), stream=stream0)
        buf828 = reinterpret_tensor(buf826, (120, 196), (1, 120), 0); del buf826  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_134.run(clone_11, buf824, convolution_17, unsqueeze_1178, buf828, 23520, 128, grid=grid(23520), stream=stream0)
        buf829 = empty((120, ), device='cuda', dtype=torch.float32)
        buf830 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_135.run(buf828, squeeze_52, buf829, buf830, 120, 196, grid=grid(120), stream=stream0)
        del buf828
        buf831 = empty_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_136.run(clone_11, buf824, convolution_17, unsqueeze_1178, buf829, squeeze_52, buf827, primals_35, buf831, 25088, 120, grid=grid(25088, 120), stream=stream0)
        del buf824
        del buf829
        del clone_11
        del convolution_17
        del primals_35
        del squeeze_52
        del unsqueeze_1178
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf832 = aten.convolution_backward(buf831, add_100, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_100
        del buf831
        del primals_194
        buf833 = buf832[0]
        buf834 = buf832[1]
        del buf832
        buf835 = empty_strided((24, 4), (1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_137.run(buf833, buf835, 96, 6272, grid=grid(96), stream=stream0)
        buf836 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_138.run(buf835, buf836, 24, 4, grid=grid(24), stream=stream0)
        buf837 = empty((24, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_139.run(buf833, convolution_16, unsqueeze_1190, buf837, 4704, 128, grid=grid(4704), stream=stream0)
        buf838 = empty((24, ), device='cuda', dtype=torch.float32)
        buf839 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_140.run(buf837, squeeze_49, buf838, buf839, 24, 196, grid=grid(24), stream=stream0)
        del buf837
        buf840 = empty((8, 24, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_141.run(buf833, convolution_16, unsqueeze_1190, buf838, squeeze_49, buf836, primals_33, buf840, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del convolution_16
        del primals_33
        del squeeze_49
        del unsqueeze_1190
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf841 = aten.convolution_backward(buf840, div_10, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_10
        del primals_193
        buf842 = buf841[0]
        buf843 = buf841[1]
        del buf841
        buf844 = empty((48, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_142.run(clone_10, buf842, buf844, 9408, 128, grid=grid(9408), stream=stream0)
        buf845 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_143.run(buf844, buf845, 48, 196, grid=grid(48), stream=stream0)
        buf846 = reinterpret_tensor(buf844, (48, 196), (1, 48), 0); del buf844  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_144.run(clone_10, buf842, convolution_15, unsqueeze_1202, buf846, 9408, 128, grid=grid(9408), stream=stream0)
        buf847 = empty((48, ), device='cuda', dtype=torch.float32)
        buf848 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_145.run(buf846, squeeze_46, buf847, buf848, 48, 196, grid=grid(48), stream=stream0)
        buf849 = empty_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_146.run(clone_10, buf842, convolution_15, unsqueeze_1202, buf847, squeeze_46, buf845, primals_31, buf849, 25088, 48, grid=grid(25088, 48), stream=stream0)
        del buf842
        del clone_10
        del convolution_15
        del primals_31
        del squeeze_46
        del unsqueeze_1202
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf850 = aten.convolution_backward(buf849, div_9, primals_192, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, False])
        del div_9
        del primals_192
        buf851 = buf850[0]
        buf852 = buf850[1]
        del buf850
        buf853 = reinterpret_tensor(buf846, (48, 196), (196, 1), 0); del buf846  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_142.run(clone_9, buf851, buf853, 9408, 128, grid=grid(9408), stream=stream0)
        buf854 = buf847; del buf847  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_143.run(buf853, buf854, 48, 196, grid=grid(48), stream=stream0)
        buf855 = reinterpret_tensor(buf853, (48, 196), (1, 48), 0); del buf853  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_144.run(clone_9, buf851, convolution_14, unsqueeze_1214, buf855, 9408, 128, grid=grid(9408), stream=stream0)
        buf856 = empty((48, ), device='cuda', dtype=torch.float32)
        buf857 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_145.run(buf855, squeeze_43, buf856, buf857, 48, 196, grid=grid(48), stream=stream0)
        buf858 = buf849; del buf849  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_146.run(clone_9, buf851, convolution_14, unsqueeze_1214, buf856, squeeze_43, buf854, primals_29, buf858, 25088, 48, grid=grid(25088, 48), stream=stream0)
        del buf851
        del clone_9
        del convolution_14
        del primals_29
        del squeeze_43
        del unsqueeze_1214
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf859 = aten.convolution_backward(buf858, add_82, primals_191, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_82
        del primals_191
        buf860 = buf859[0]
        buf861 = buf859[1]
        del buf859
        buf862 = buf835; del buf835  # reuse
        buf864 = empty_strided((24, 4), (1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_147.run(buf833, buf860, convolution_13, unsqueeze_1226, buf862, buf864, 96, 6272, grid=grid(96), stream=stream0)
        buf863 = buf838; del buf838  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_138.run(buf862, buf863, 24, 4, grid=grid(24), stream=stream0)
        buf865 = empty((24, ), device='cuda', dtype=torch.float32)
        buf866 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_148.run(buf864, squeeze_40, buf865, buf866, 24, 4, grid=grid(24), stream=stream0)
        buf867 = buf840; del buf840  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_149.run(buf833, buf860, convolution_13, unsqueeze_1226, buf865, squeeze_40, buf863, primals_27, buf867, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del convolution_13
        del primals_27
        del squeeze_40
        del unsqueeze_1226
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf868 = aten.convolution_backward(buf867, div_8, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_8
        del primals_190
        buf869 = buf868[0]
        buf870 = buf868[1]
        del buf868
        buf871 = reinterpret_tensor(buf855, (48, 196), (196, 1), 0); del buf855  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_142.run(clone_8, buf869, buf871, 9408, 128, grid=grid(9408), stream=stream0)
        buf872 = buf856; del buf856  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_143.run(buf871, buf872, 48, 196, grid=grid(48), stream=stream0)
        buf873 = reinterpret_tensor(buf871, (48, 196), (1, 48), 0); del buf871  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_144.run(clone_8, buf869, convolution_12, unsqueeze_1238, buf873, 9408, 128, grid=grid(9408), stream=stream0)
        buf874 = empty((48, ), device='cuda', dtype=torch.float32)
        buf875 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_145.run(buf873, squeeze_37, buf874, buf875, 48, 196, grid=grid(48), stream=stream0)
        buf876 = buf858; del buf858  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_146.run(clone_8, buf869, convolution_12, unsqueeze_1238, buf874, squeeze_37, buf872, primals_25, buf876, 25088, 48, grid=grid(25088, 48), stream=stream0)
        del buf869
        del clone_8
        del convolution_12
        del primals_25
        del squeeze_37
        del unsqueeze_1238
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf877 = aten.convolution_backward(buf876, div_7, primals_189, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, False])
        del div_7
        del primals_189
        buf878 = buf877[0]
        buf879 = buf877[1]
        del buf877
        buf880 = reinterpret_tensor(buf873, (48, 196), (196, 1), 0); del buf873  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_142.run(clone_7, buf878, buf880, 9408, 128, grid=grid(9408), stream=stream0)
        buf881 = buf874; del buf874  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_143.run(buf880, buf881, 48, 196, grid=grid(48), stream=stream0)
        buf882 = reinterpret_tensor(buf880, (48, 196), (1, 48), 0); del buf880  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_144.run(clone_7, buf878, convolution_11, unsqueeze_1250, buf882, 9408, 128, grid=grid(9408), stream=stream0)
        buf883 = empty((48, ), device='cuda', dtype=torch.float32)
        buf884 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_145.run(buf882, squeeze_34, buf883, buf884, 48, 196, grid=grid(48), stream=stream0)
        buf885 = buf876; del buf876  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_146.run(clone_7, buf878, convolution_11, unsqueeze_1250, buf883, squeeze_34, buf881, primals_23, buf885, 25088, 48, grid=grid(25088, 48), stream=stream0)
        del buf878
        del clone_7
        del convolution_11
        del primals_23
        del squeeze_34
        del unsqueeze_1250
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf886 = aten.convolution_backward(buf885, add_64, primals_188, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_64
        del primals_188
        buf887 = buf886[0]
        buf888 = buf886[1]
        del buf886
        buf889 = buf864; del buf864  # reuse
        buf891 = buf862; del buf862  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_150.run(buf833, buf860, buf887, convolution_10, unsqueeze_1262, buf889, buf891, 96, 6272, grid=grid(96), stream=stream0)
        buf890 = buf865; del buf865  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_138.run(buf889, buf890, 24, 4, grid=grid(24), stream=stream0)
        buf892 = empty((24, ), device='cuda', dtype=torch.float32)
        buf894 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_148.run(buf891, squeeze_31, buf892, buf894, 24, 4, grid=grid(24), stream=stream0)
        buf893 = buf867; del buf867  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_151.run(buf833, buf860, buf887, convolution_10, unsqueeze_1262, buf892, squeeze_31, buf890, primals_21, buf893, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del convolution_10
        del primals_21
        del squeeze_31
        del unsqueeze_1262
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf895 = aten.convolution_backward(buf893, div_6, primals_187, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf893
        del div_6
        del primals_187
        buf896 = buf895[0]
        buf897 = buf895[1]
        del buf895
        buf898 = reinterpret_tensor(buf882, (48, 196), (196, 1), 0); del buf882  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_142.run(clone_6, buf896, buf898, 9408, 128, grid=grid(9408), stream=stream0)
        buf899 = buf883; del buf883  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_143.run(buf898, buf899, 48, 196, grid=grid(48), stream=stream0)
        buf900 = reinterpret_tensor(buf898, (48, 196), (1, 48), 0); del buf898  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_144.run(clone_6, buf896, convolution_9, unsqueeze_1274, buf900, 9408, 128, grid=grid(9408), stream=stream0)
        buf901 = empty((48, ), device='cuda', dtype=torch.float32)
        buf902 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_145.run(buf900, squeeze_28, buf901, buf902, 48, 196, grid=grid(48), stream=stream0)
        buf903 = buf885; del buf885  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_146.run(clone_6, buf896, convolution_9, unsqueeze_1274, buf901, squeeze_28, buf899, primals_19, buf903, 25088, 48, grid=grid(25088, 48), stream=stream0)
        del buf896
        del clone_6
        del convolution_9
        del primals_19
        del squeeze_28
        del unsqueeze_1274
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf904 = aten.convolution_backward(buf903, div_5, primals_186, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, False])
        del div_5
        del primals_186
        buf905 = buf904[0]
        buf906 = buf904[1]
        del buf904
        buf907 = reinterpret_tensor(buf900, (48, 196), (196, 1), 0); del buf900  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_142.run(clone_5, buf905, buf907, 9408, 128, grid=grid(9408), stream=stream0)
        buf908 = buf901; del buf901  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_143.run(buf907, buf908, 48, 196, grid=grid(48), stream=stream0)
        buf909 = reinterpret_tensor(buf907, (48, 196), (1, 48), 0); del buf907  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_144.run(clone_5, buf905, convolution_8, unsqueeze_1286, buf909, 9408, 128, grid=grid(9408), stream=stream0)
        buf910 = empty((48, ), device='cuda', dtype=torch.float32)
        buf911 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_145.run(buf909, squeeze_25, buf910, buf911, 48, 196, grid=grid(48), stream=stream0)
        del buf909
        buf912 = buf903; del buf903  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_146.run(clone_5, buf905, convolution_8, unsqueeze_1286, buf910, squeeze_25, buf908, primals_17, buf912, 25088, 48, grid=grid(25088, 48), stream=stream0)
        del buf905
        del buf910
        del clone_5
        del convolution_8
        del primals_17
        del squeeze_25
        del unsqueeze_1286
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf913 = aten.convolution_backward(buf912, add_46, primals_185, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_46
        del buf912
        del primals_185
        buf914 = buf913[0]
        buf915 = buf913[1]
        del buf913
        buf916 = buf891; del buf891  # reuse
        buf918 = buf889; del buf889  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_152.run(buf833, buf860, buf887, buf914, convolution_7, unsqueeze_1298, buf916, buf918, 96, 6272, grid=grid(96), stream=stream0)
        buf917 = buf892; del buf892  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_138.run(buf916, buf917, 24, 4, grid=grid(24), stream=stream0)
        del buf916
        buf919 = empty((24, ), device='cuda', dtype=torch.float32)
        buf921 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_148.run(buf918, squeeze_22, buf919, buf921, 24, 4, grid=grid(24), stream=stream0)
        del buf918
        buf920 = buf833; del buf833  # reuse
        buf922 = buf920; del buf920  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_153.run(buf922, buf860, buf887, buf914, convolution_7, unsqueeze_1298, buf919, squeeze_22, buf917, primals_15, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del buf860
        del buf887
        del buf914
        del buf919
        del convolution_7
        del primals_15
        del squeeze_22
        del unsqueeze_1298
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf923 = aten.convolution_backward(buf922, div_4, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf922
        del div_4
        del primals_184
        buf924 = buf923[0]
        buf925 = buf923[1]
        del buf923
        buf926 = empty((64, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_154.run(clone_4, buf924, buf926, 12544, 128, grid=grid(12544), stream=stream0)
        buf927 = reinterpret_tensor(buf811, (64, ), (1, ), 0); del buf811  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_155.run(buf926, buf927, 64, 196, grid=grid(64), stream=stream0)
        buf928 = reinterpret_tensor(buf926, (64, 196), (1, 64), 0); del buf926  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_156.run(clone_4, buf924, convolution_6, unsqueeze_1310, buf928, 12544, 128, grid=grid(12544), stream=stream0)
        buf929 = empty((64, ), device='cuda', dtype=torch.float32)
        buf930 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_157.run(buf928, squeeze_19, buf929, buf930, 64, 196, grid=grid(64), stream=stream0)
        buf931 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_158.run(clone_4, buf924, convolution_6, unsqueeze_1310, buf929, squeeze_19, buf927, primals_13, buf931, 25088, 64, grid=grid(25088, 64), stream=stream0)
        del buf924
        del clone_4
        del convolution_6
        del primals_13
        del squeeze_19
        del unsqueeze_1310
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf932 = aten.convolution_backward(buf931, div_3, primals_183, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 64, [True, True, False])
        del div_3
        del primals_183
        buf933 = buf932[0]
        buf934 = buf932[1]
        del buf932
        buf935 = empty((64, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_159.run(clone_3, buf933, buf935, 50176, 128, grid=grid(50176), stream=stream0)
        buf936 = buf929; del buf929  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_160.run(buf935, buf936, 64, 784, grid=grid(64), stream=stream0)
        buf937 = reinterpret_tensor(buf935, (64, 784), (1, 64), 0); del buf935  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_161.run(clone_3, buf933, convolution_5, unsqueeze_1322, buf937, 50176, 128, grid=grid(50176), stream=stream0)
        buf938 = empty((64, ), device='cuda', dtype=torch.float32)
        buf939 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_162.run(buf937, squeeze_16, buf938, buf939, 64, 784, grid=grid(64), stream=stream0)
        del buf937
        buf940 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_163.run(clone_3, buf933, convolution_5, unsqueeze_1322, buf938, squeeze_16, buf936, primals_11, buf940, 100352, 64, grid=grid(100352, 64), stream=stream0)
        del buf933
        del buf938
        del clone_3
        del convolution_5
        del primals_11
        del squeeze_16
        del unsqueeze_1322
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf941 = aten.convolution_backward(buf940, add_29, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_29
        del buf940
        del primals_182
        buf942 = buf941[0]
        buf943 = buf941[1]
        del buf941
        buf944 = empty((16, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_164.run(buf942, buf944, 208, 7720, grid=grid(208), stream=stream0)
        buf945 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_165.run(buf944, buf945, 16, 13, grid=grid(16), stream=stream0)
        buf946 = reinterpret_tensor(buf928, (16, 784), (784, 1), 0); del buf928  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_166.run(buf942, convolution_4, unsqueeze_1334, buf946, 12544, 128, grid=grid(12544), stream=stream0)
        buf947 = empty((16, ), device='cuda', dtype=torch.float32)
        buf948 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_167.run(buf946, squeeze_13, buf947, buf948, 16, 784, grid=grid(16), stream=stream0)
        buf949 = reinterpret_tensor(buf931, (8, 16, 112, 112), (200704, 12544, 112, 1), 0); del buf931  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_168.run(buf942, convolution_4, unsqueeze_1334, buf947, squeeze_13, buf945, primals_9, buf949, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del convolution_4
        del primals_9
        del squeeze_13
        del unsqueeze_1334
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf950 = aten.convolution_backward(buf949, div_2, primals_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_2
        del primals_181
        buf951 = buf950[0]
        buf952 = buf950[1]
        del buf950
        buf953 = buf946; del buf946  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_169.run(clone_2, buf951, buf953, 12544, 128, grid=grid(12544), stream=stream0)
        buf954 = buf947; del buf947  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_170.run(buf953, buf954, 16, 784, grid=grid(16), stream=stream0)
        buf955 = reinterpret_tensor(buf953, (16, 784), (1, 16), 0); del buf953  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_171.run(clone_2, buf951, convolution_3, unsqueeze_1346, buf955, 12544, 128, grid=grid(12544), stream=stream0)
        buf956 = empty((16, ), device='cuda', dtype=torch.float32)
        buf957 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_172.run(buf955, squeeze_10, buf956, buf957, 16, 784, grid=grid(16), stream=stream0)
        buf958 = reinterpret_tensor(buf949, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf949  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_173.run(clone_2, buf951, convolution_3, unsqueeze_1346, buf956, squeeze_10, buf954, primals_7, buf958, 100352, 16, grid=grid(100352, 16), stream=stream0)
        del buf951
        del clone_2
        del convolution_3
        del primals_7
        del squeeze_10
        del unsqueeze_1346
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf959 = aten.convolution_backward(buf958, add_17, primals_180, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False])
        del add_17
        del primals_180
        buf960 = buf959[0]
        buf961 = buf959[1]
        del buf959
        buf962 = buf944; del buf944  # reuse
        buf964 = empty((16, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_174.run(buf942, buf960, convolution_2, unsqueeze_1358, buf962, buf964, 208, 7720, grid=grid(208), stream=stream0)
        buf963 = buf956; del buf956  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_165.run(buf962, buf963, 16, 13, grid=grid(16), stream=stream0)
        buf965 = empty((16, ), device='cuda', dtype=torch.float32)
        buf966 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_175.run(buf964, squeeze_7, buf965, buf966, 16, 13, grid=grid(16), stream=stream0)
        buf967 = reinterpret_tensor(buf958, (8, 16, 112, 112), (200704, 12544, 112, 1), 0); del buf958  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_176.run(buf942, buf960, convolution_2, unsqueeze_1358, buf965, squeeze_7, buf963, primals_5, buf967, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del convolution_2
        del primals_5
        del squeeze_7
        del unsqueeze_1358
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf968 = aten.convolution_backward(buf967, div_1, primals_179, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_1
        del primals_179
        buf969 = buf968[0]
        buf970 = buf968[1]
        del buf968
        buf971 = reinterpret_tensor(buf955, (16, 784), (784, 1), 0); del buf955  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_169.run(clone_1, buf969, buf971, 12544, 128, grid=grid(12544), stream=stream0)
        buf972 = buf965; del buf965  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_170.run(buf971, buf972, 16, 784, grid=grid(16), stream=stream0)
        buf973 = reinterpret_tensor(buf971, (16, 784), (1, 16), 0); del buf971  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_171.run(clone_1, buf969, convolution_1, unsqueeze_1370, buf973, 12544, 128, grid=grid(12544), stream=stream0)
        buf974 = empty((16, ), device='cuda', dtype=torch.float32)
        buf975 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_172.run(buf973, squeeze_4, buf974, buf975, 16, 784, grid=grid(16), stream=stream0)
        del buf973
        buf976 = reinterpret_tensor(buf967, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf967  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_173.run(clone_1, buf969, convolution_1, unsqueeze_1370, buf974, squeeze_4, buf972, primals_3, buf976, 100352, 16, grid=grid(100352, 16), stream=stream0)
        del buf969
        del clone_1
        del convolution_1
        del primals_3
        del squeeze_4
        del unsqueeze_1370
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf977 = aten.convolution_backward(buf976, div, primals_178, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False])
        del div
        del primals_178
        buf978 = buf977[0]
        buf979 = buf977[1]
        del buf977
        buf980 = buf964; del buf964  # reuse
        buf982 = buf962; del buf962  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_hardswish_backward_native_batch_norm_backward_177.run(clone, buf942, buf960, buf978, convolution, unsqueeze_1382, buf980, buf982, 208, 7720, grid=grid(208), stream=stream0)
        buf981 = buf974; del buf974  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_165.run(buf980, buf981, 16, 13, grid=grid(16), stream=stream0)
        del buf980
        buf983 = empty((16, ), device='cuda', dtype=torch.float32)
        buf985 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_175.run(buf982, squeeze_1, buf983, buf985, 16, 13, grid=grid(16), stream=stream0)
        del buf982
        buf984 = buf942; del buf942  # reuse
        buf986 = buf976; del buf976  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_hardswish_backward_native_batch_norm_backward_178.run(buf984, clone, buf960, buf978, convolution, unsqueeze_1382, buf983, squeeze_1, buf981, primals_1, buf986, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del buf960
        del buf978
        del buf983
        del buf984
        del clone
        del convolution
        del primals_1
        del squeeze_1
        del unsqueeze_1382
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf987 = aten.convolution_backward(buf986, primals_598, primals_177, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf986
        del primals_177
        del primals_598
        buf988 = buf987[1]
        return (buf985, buf981, buf975, buf972, buf966, buf963, buf957, buf954, buf948, buf945, buf939, buf936, buf930, buf927, buf921, buf917, buf911, buf908, buf902, buf899, buf894, buf890, buf884, buf881, buf875, buf872, buf866, buf863, buf857, buf854, buf848, buf845, buf839, buf836, buf830, buf827, buf821, buf817, buf799, buf796, buf790, buf787, buf781, buf777, buf759, buf756, buf751, buf748, buf742, buf738, buf721, buf718, buf713, buf710, buf704, buf700, buf682, buf680, buf675, buf672, buf666, buf662, buf644, buf641, buf636, buf633, buf627, buf624, buf618, buf615, buf609, buf606, buf600, buf597, buf591, buf588, buf583, buf580, buf574, buf571, buf566, buf563, buf558, buf555, buf549, buf546, buf540, buf538, buf533, buf530, buf524, buf521, buf515, buf512, buf507, buf504, buf498, buf494, buf476, buf474, buf469, buf466, buf460, buf456, buf438, buf435, buf429, buf426, buf420, buf416, buf398, buf395, buf390, buf387, buf381, buf377, buf360, buf357, buf352, buf349, buf343, buf339, buf321, buf319, buf314, buf311, buf305, buf301, buf283, buf280, buf275, buf272, buf266, buf262, buf245, buf243, buf238, buf235, buf229, buf225, buf208, buf205, buf199, buf196, buf190, buf186, buf169, buf166, buf161, buf158, buf152, buf148, buf132, buf129, buf124, buf121, buf115, buf111, buf94, buf92, buf87, buf84, buf78, buf74, buf57, buf54, buf49, buf46, buf40, buf36, buf19, buf16, buf11, buf8, reinterpret_tensor(buf1, (1000, 1984), (1984, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), buf988, buf979, buf970, buf961, buf952, buf943, buf934, buf925, buf915, buf906, buf897, buf888, buf879, buf870, buf861, buf852, buf843, buf834, buf825, buf815, buf812, buf810, buf807, buf803, buf794, buf785, buf775, buf772, buf770, buf767, buf763, buf755, buf746, buf736, buf733, buf731, buf728, buf724, buf717, buf708, buf698, buf695, buf693, buf690, buf686, buf679, buf670, buf660, buf657, buf655, buf652, buf648, buf640, buf631, buf622, buf613, buf604, buf595, buf587, buf578, buf569, buf562, buf553, buf544, buf537, buf528, buf519, buf511, buf502, buf492, buf489, buf487, buf484, buf480, buf473, buf464, buf454, buf451, buf449, buf446, buf442, buf433, buf424, buf414, buf411, buf409, buf406, buf402, buf394, buf385, buf375, buf372, buf370, buf367, buf363, buf356, buf347, buf337, buf334, buf332, buf329, buf325, buf318, buf309, buf299, buf296, buf294, buf291, buf287, buf279, buf270, buf260, buf257, buf255, buf252, buf249, buf242, buf233, buf223, buf220, buf218, buf215, buf212, buf203, buf194, buf184, buf181, buf179, buf176, buf173, buf165, buf156, buf146, buf143, buf141, buf138, buf135, buf128, buf119, buf109, buf106, buf104, buf101, buf98, buf91, buf82, buf72, buf69, buf67, buf64, buf61, buf53, buf44, buf34, buf31, buf29, buf26, buf23, buf15, buf6, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((120, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((8, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((120, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((200, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((200, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((72, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((360, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((360, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((24, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((360, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((720, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((720, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((32, 720, 1, 1), (720, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((720, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((184, 720, 1, 1), (720, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((1104, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((48, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((1104, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((224, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((1984, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    div = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_1 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_17 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_2 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_29 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_3 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_4 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_46 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_5 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    div_5 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_6 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    div_6 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_64 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_7 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_8 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_82 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_9 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_10 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_100 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_11 = rand_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_12 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    mul_147 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_119 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_14 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_15 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    mean_1 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    mul_172 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_139 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_17 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_18 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    div_20 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    mean_2 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    mul_197 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_159 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_20 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_21 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    mean_3 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    mul_222 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_179 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_23 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_24 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    mean_4 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    div_29 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    mul_247 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_199 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 200, 28, 28), (156800, 1, 5600, 200), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_26 = rand_strided((8, 200, 28, 28), (156800, 1, 5600, 200), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((8, 200, 28, 28), (156800, 1, 5600, 200), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_27 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda:0', dtype=torch.float32)
    div_32 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_216 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_28 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_29 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_234 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_30 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    div_35 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_31 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_252 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_32 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_33 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    div_38 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_270 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_34 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_35 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_288 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_36 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    div_41 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_37 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    mean_5 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda:0', dtype=torch.float32)
    convolution_59 = rand_strided((8, 24, 1, 1), (24, 1, 24, 24), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((8, 24, 1, 1), (24, 1, 24, 24), device='cuda:0', dtype=torch.float32)
    div_44 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda:0', dtype=torch.float32)
    mul_387 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_307 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_39 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    convolution_63 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_40 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    mean_6 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda:0', dtype=torch.float32)
    convolution_64 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    div_47 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda:0', dtype=torch.float32)
    mul_412 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    convolution_66 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_327 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda:0', dtype=torch.float32)
    convolution_67 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    squeeze_160 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_42 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    convolution_68 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    squeeze_163 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_43 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    div_50 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    mean_7 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda:0', dtype=torch.float32)
    convolution_69 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda:0', dtype=torch.float32)
    mul_437 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    convolution_71 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda:0', dtype=torch.float32)
    squeeze_166 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_347 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda:0', dtype=torch.float32)
    convolution_72 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    squeeze_169 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_45 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    div_53 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    convolution_73 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    squeeze_172 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_46 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    mean_8 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda:0', dtype=torch.float32)
    convolution_74 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    div_56 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda:0', dtype=torch.float32)
    mul_462 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    convolution_76 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda:0', dtype=torch.float32)
    squeeze_175 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_367 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda:0', dtype=torch.float32)
    convolution_77 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    squeeze_178 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_48 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    convolution_78 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    squeeze_181 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_49 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    mean_9 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda:0', dtype=torch.float32)
    convolution_79 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    div_59 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda:0', dtype=torch.float32)
    mul_487 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    convolution_81 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda:0', dtype=torch.float32)
    squeeze_184 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_387 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda:0', dtype=torch.float32)
    convolution_82 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    squeeze_187 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_51 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    convolution_83 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    squeeze_190 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_52 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    div_62 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    mean_10 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda:0', dtype=torch.float32)
    convolution_84 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    div_63 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    div_64 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda:0', dtype=torch.float32)
    mul_512 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda:0', dtype=torch.float32)
    convolution_86 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda:0', dtype=torch.float32)
    squeeze_193 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_407 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda:0', dtype=torch.float32)
    convolution_87 = rand_strided((8, 720, 14, 14), (141120, 1, 10080, 720), device='cuda:0', dtype=torch.float32)
    squeeze_196 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_54 = rand_strided((8, 720, 14, 14), (141120, 1, 10080, 720), device='cuda:0', dtype=torch.float32)
    div_65 = rand_strided((8, 720, 14, 14), (141120, 1, 10080, 720), device='cuda:0', dtype=torch.float32)
    convolution_88 = rand_strided((8, 720, 7, 7), (35280, 1, 5040, 720), device='cuda:0', dtype=torch.float32)
    squeeze_199 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_55 = rand_strided((8, 720, 7, 7), (35280, 1, 5040, 720), device='cuda:0', dtype=torch.float32)
    div_66 = rand_strided((8, 720, 7, 7), (35280, 1, 5040, 720), device='cuda:0', dtype=torch.float32)
    mean_11 = rand_strided((8, 720, 1, 1), (720, 1, 720, 720), device='cuda:0', dtype=torch.float32)
    convolution_89 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    div_67 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    div_68 = rand_strided((8, 720, 1, 1), (720, 1, 720, 720), device='cuda:0', dtype=torch.float32)
    mul_537 = rand_strided((8, 720, 7, 7), (35280, 1, 5040, 720), device='cuda:0', dtype=torch.float32)
    convolution_91 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda:0', dtype=torch.float32)
    squeeze_202 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_426 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda:0', dtype=torch.float32)
    convolution_92 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    squeeze_205 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_57 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    div_69 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    convolution_93 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    squeeze_208 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_58 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    div_70 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    mean_12 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda:0', dtype=torch.float32)
    convolution_94 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    div_71 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    div_72 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda:0', dtype=torch.float32)
    mul_562 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    convolution_96 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda:0', dtype=torch.float32)
    squeeze_211 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_446 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda:0', dtype=torch.float32)
    convolution_97 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    squeeze_214 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_60 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    div_73 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    convolution_98 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    squeeze_217 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_61 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    div_74 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    mean_13 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda:0', dtype=torch.float32)
    convolution_99 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    div_75 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    div_76 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda:0', dtype=torch.float32)
    mul_587 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    convolution_101 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda:0', dtype=torch.float32)
    squeeze_220 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_466 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda:0', dtype=torch.float32)
    convolution_102 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    squeeze_223 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_63 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    div_77 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    convolution_103 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    squeeze_226 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_64 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    div_78 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    mean_14 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda:0', dtype=torch.float32)
    convolution_104 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    div_79 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    div_80 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda:0', dtype=torch.float32)
    mul_612 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    convolution_106 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda:0', dtype=torch.float32)
    squeeze_229 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_486 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda:0', dtype=torch.float32)
    convolution_107 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    squeeze_232 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_66 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    div_81 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    convolution_108 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    squeeze_235 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_67 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    div_82 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    mean_15 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda:0', dtype=torch.float32)
    convolution_109 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    div_83 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    div_84 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda:0', dtype=torch.float32)
    mul_637 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    convolution_111 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda:0', dtype=torch.float32)
    squeeze_238 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_506 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda:0', dtype=torch.float32)
    convolution_112 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    squeeze_241 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_69 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    div_85 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    convolution_113 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    squeeze_244 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_70 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    div_86 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    mean_16 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda:0', dtype=torch.float32)
    convolution_114 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    div_87 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    div_88 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda:0', dtype=torch.float32)
    mul_662 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    convolution_116 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda:0', dtype=torch.float32)
    squeeze_247 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_526 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda:0', dtype=torch.float32)
    convolution_117 = rand_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda:0', dtype=torch.float32)
    squeeze_250 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_72 = rand_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda:0', dtype=torch.float32)
    div_89 = rand_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda:0', dtype=torch.float32)
    convolution_118 = rand_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda:0', dtype=torch.float32)
    squeeze_253 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_73 = rand_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda:0', dtype=torch.float32)
    div_90 = rand_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda:0', dtype=torch.float32)
    mean_17 = rand_strided((8, 1104, 1, 1), (1104, 1, 1104, 1104), device='cuda:0', dtype=torch.float32)
    convolution_119 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    div_91 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda:0', dtype=torch.float32)
    div_92 = rand_strided((8, 1104, 1, 1), (1104, 1, 1104, 1104), device='cuda:0', dtype=torch.float32)
    mul_687 = rand_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda:0', dtype=torch.float32)
    convolution_121 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda:0', dtype=torch.float32)
    squeeze_256 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_545 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda:0', dtype=torch.float32)
    convolution_122 = rand_strided((8, 1344, 7, 7), (65856, 1, 9408, 1344), device='cuda:0', dtype=torch.float32)
    squeeze_259 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_75 = rand_strided((8, 1344, 7, 7), (65856, 1, 9408, 1344), device='cuda:0', dtype=torch.float32)
    mean_18 = rand_strided((8, 1344, 1, 1), (1344, 1, 1344, 1344), device='cuda:0', dtype=torch.float32)
    convolution_123 = rand_strided((8, 1984, 1, 1), (1984, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((8, 1984), (1984, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1984), (1984, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_350 = rand_strided((1, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_362 = rand_strided((1, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and = rand_strided((8, 1104, 1, 1), (1104, 1, 1104, 1104), device='cuda:0', dtype=torch.bool)
    unsqueeze_374 = rand_strided((1, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_386 = rand_strided((1, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_398 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_1 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda:0', dtype=torch.bool)
    unsqueeze_410 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_422 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_434 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_2 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda:0', dtype=torch.bool)
    unsqueeze_446 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_458 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_470 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_3 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda:0', dtype=torch.bool)
    unsqueeze_482 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_494 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_506 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_4 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda:0', dtype=torch.bool)
    unsqueeze_518 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_530 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_542 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_5 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda:0', dtype=torch.bool)
    unsqueeze_554 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_566 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_578 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_6 = rand_strided((8, 720, 1, 1), (720, 1, 720, 720), device='cuda:0', dtype=torch.bool)
    unsqueeze_590 = rand_strided((1, 720, 1, 1), (720, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_602 = rand_strided((1, 720, 1, 1), (720, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_614 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_7 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda:0', dtype=torch.bool)
    unsqueeze_626 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_638 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_650 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_8 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda:0', dtype=torch.bool)
    unsqueeze_662 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_674 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_686 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_9 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda:0', dtype=torch.bool)
    unsqueeze_698 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_710 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_722 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_10 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda:0', dtype=torch.bool)
    unsqueeze_734 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_746 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_758 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_11 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda:0', dtype=torch.bool)
    unsqueeze_770 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_782 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_794 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_12 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda:0', dtype=torch.bool)
    unsqueeze_806 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_818 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_830 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_842 = rand_strided((1, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_854 = rand_strided((1, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_866 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_878 = rand_strided((1, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_890 = rand_strided((1, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_902 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_914 = rand_strided((1, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_926 = rand_strided((1, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_938 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_950 = rand_strided((1, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_962 = rand_strided((1, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_974 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_986 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_998 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1010 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_13 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.bool)
    unsqueeze_1022 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1034 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1046 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_14 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.bool)
    unsqueeze_1058 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1070 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1082 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_15 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.bool)
    unsqueeze_1094 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1106 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1118 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_16 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.bool)
    unsqueeze_1130 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1142 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1154 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_17 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.bool)
    unsqueeze_1166 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1178 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1190 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1202 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1214 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1226 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1238 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1250 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1262 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1274 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1286 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1298 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1310 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1322 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1334 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1346 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1358 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1370 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1382 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_198, primals_200, primals_201, primals_202, primals_203, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_214, primals_215, primals_216, primals_217, primals_219, primals_221, primals_222, primals_223, primals_224, primals_226, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_248, primals_250, primals_251, primals_252, primals_253, primals_255, primals_257, primals_258, primals_259, primals_260, primals_262, primals_264, primals_265, primals_266, primals_267, primals_269, primals_271, primals_272, primals_273, primals_274, primals_276, primals_278, primals_279, primals_280, primals_281, primals_283, primals_285, primals_286, primals_287, primals_288, primals_290, primals_292, primals_293, primals_294, primals_295, primals_297, primals_299, primals_300, primals_301, primals_302, primals_304, primals_306, primals_307, primals_308, primals_309, primals_311, primals_313, primals_314, primals_315, primals_316, primals_318, primals_320, primals_321, primals_322, primals_323, primals_325, primals_327, primals_328, primals_329, primals_330, primals_332, primals_334, primals_335, primals_336, primals_598, convolution, squeeze_1, clone, div, convolution_1, squeeze_4, clone_1, div_1, convolution_2, squeeze_7, add_17, convolution_3, squeeze_10, clone_2, div_2, convolution_4, squeeze_13, add_29, convolution_5, squeeze_16, clone_3, div_3, convolution_6, squeeze_19, clone_4, div_4, convolution_7, squeeze_22, add_46, convolution_8, squeeze_25, clone_5, div_5, convolution_9, squeeze_28, clone_6, div_6, convolution_10, squeeze_31, add_64, convolution_11, squeeze_34, clone_7, div_7, convolution_12, squeeze_37, clone_8, div_8, convolution_13, squeeze_40, add_82, convolution_14, squeeze_43, clone_9, div_9, convolution_15, squeeze_46, clone_10, div_10, convolution_16, squeeze_49, add_100, convolution_17, squeeze_52, clone_11, div_11, convolution_18, squeeze_55, clone_12, div_12, mean, convolution_19, div_13, div_14, mul_147, convolution_21, squeeze_58, add_119, convolution_22, squeeze_61, clone_14, div_15, convolution_23, squeeze_64, clone_15, div_16, mean_1, convolution_24, div_17, div_18, mul_172, convolution_26, squeeze_67, add_139, convolution_27, squeeze_70, clone_17, div_19, convolution_28, squeeze_73, clone_18, div_20, mean_2, convolution_29, div_21, div_22, mul_197, convolution_31, squeeze_76, add_159, convolution_32, squeeze_79, clone_20, div_23, convolution_33, squeeze_82, clone_21, div_24, mean_3, convolution_34, div_25, div_26, mul_222, convolution_36, squeeze_85, add_179, convolution_37, squeeze_88, clone_23, div_27, convolution_38, squeeze_91, clone_24, div_28, mean_4, convolution_39, div_29, div_30, mul_247, convolution_41, squeeze_94, add_199, convolution_42, squeeze_97, clone_26, div_31, convolution_43, squeeze_100, clone_27, div_32, convolution_44, squeeze_103, add_216, convolution_45, squeeze_106, clone_28, div_33, convolution_46, squeeze_109, clone_29, div_34, convolution_47, squeeze_112, add_234, convolution_48, squeeze_115, clone_30, div_35, convolution_49, squeeze_118, clone_31, div_36, convolution_50, squeeze_121, add_252, convolution_51, squeeze_124, clone_32, div_37, convolution_52, squeeze_127, clone_33, div_38, convolution_53, squeeze_130, add_270, convolution_54, squeeze_133, clone_34, div_39, convolution_55, squeeze_136, clone_35, div_40, convolution_56, squeeze_139, add_288, convolution_57, squeeze_142, clone_36, div_41, convolution_58, squeeze_145, clone_37, div_42, mean_5, convolution_59, div_43, div_44, mul_387, convolution_61, squeeze_148, add_307, convolution_62, squeeze_151, clone_39, div_45, convolution_63, squeeze_154, clone_40, div_46, mean_6, convolution_64, div_47, div_48, mul_412, convolution_66, squeeze_157, add_327, convolution_67, squeeze_160, clone_42, div_49, convolution_68, squeeze_163, clone_43, div_50, mean_7, convolution_69, div_51, div_52, mul_437, convolution_71, squeeze_166, add_347, convolution_72, squeeze_169, clone_45, div_53, convolution_73, squeeze_172, clone_46, div_54, mean_8, convolution_74, div_55, div_56, mul_462, convolution_76, squeeze_175, add_367, convolution_77, squeeze_178, clone_48, div_57, convolution_78, squeeze_181, clone_49, div_58, mean_9, convolution_79, div_59, div_60, mul_487, convolution_81, squeeze_184, add_387, convolution_82, squeeze_187, clone_51, div_61, convolution_83, squeeze_190, clone_52, div_62, mean_10, convolution_84, div_63, div_64, mul_512, convolution_86, squeeze_193, add_407, convolution_87, squeeze_196, clone_54, div_65, convolution_88, squeeze_199, clone_55, div_66, mean_11, convolution_89, div_67, div_68, mul_537, convolution_91, squeeze_202, add_426, convolution_92, squeeze_205, clone_57, div_69, convolution_93, squeeze_208, clone_58, div_70, mean_12, convolution_94, div_71, div_72, mul_562, convolution_96, squeeze_211, add_446, convolution_97, squeeze_214, clone_60, div_73, convolution_98, squeeze_217, clone_61, div_74, mean_13, convolution_99, div_75, div_76, mul_587, convolution_101, squeeze_220, add_466, convolution_102, squeeze_223, clone_63, div_77, convolution_103, squeeze_226, clone_64, div_78, mean_14, convolution_104, div_79, div_80, mul_612, convolution_106, squeeze_229, add_486, convolution_107, squeeze_232, clone_66, div_81, convolution_108, squeeze_235, clone_67, div_82, mean_15, convolution_109, div_83, div_84, mul_637, convolution_111, squeeze_238, add_506, convolution_112, squeeze_241, clone_69, div_85, convolution_113, squeeze_244, clone_70, div_86, mean_16, convolution_114, div_87, div_88, mul_662, convolution_116, squeeze_247, add_526, convolution_117, squeeze_250, clone_72, div_89, convolution_118, squeeze_253, clone_73, div_90, mean_17, convolution_119, div_91, div_92, mul_687, convolution_121, squeeze_256, add_545, convolution_122, squeeze_259, clone_75, mean_18, convolution_123, view_1, permute_1, unsqueeze_350, unsqueeze_362, bitwise_and, unsqueeze_374, unsqueeze_386, unsqueeze_398, bitwise_and_1, unsqueeze_410, unsqueeze_422, unsqueeze_434, bitwise_and_2, unsqueeze_446, unsqueeze_458, unsqueeze_470, bitwise_and_3, unsqueeze_482, unsqueeze_494, unsqueeze_506, bitwise_and_4, unsqueeze_518, unsqueeze_530, unsqueeze_542, bitwise_and_5, unsqueeze_554, unsqueeze_566, unsqueeze_578, bitwise_and_6, unsqueeze_590, unsqueeze_602, unsqueeze_614, bitwise_and_7, unsqueeze_626, unsqueeze_638, unsqueeze_650, bitwise_and_8, unsqueeze_662, unsqueeze_674, unsqueeze_686, bitwise_and_9, unsqueeze_698, unsqueeze_710, unsqueeze_722, bitwise_and_10, unsqueeze_734, unsqueeze_746, unsqueeze_758, bitwise_and_11, unsqueeze_770, unsqueeze_782, unsqueeze_794, bitwise_and_12, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, unsqueeze_914, unsqueeze_926, unsqueeze_938, unsqueeze_950, unsqueeze_962, unsqueeze_974, unsqueeze_986, unsqueeze_998, unsqueeze_1010, bitwise_and_13, unsqueeze_1022, unsqueeze_1034, unsqueeze_1046, bitwise_and_14, unsqueeze_1058, unsqueeze_1070, unsqueeze_1082, bitwise_and_15, unsqueeze_1094, unsqueeze_1106, unsqueeze_1118, bitwise_and_16, unsqueeze_1130, unsqueeze_1142, unsqueeze_1154, bitwise_and_17, unsqueeze_1166, unsqueeze_1178, unsqueeze_1190, unsqueeze_1202, unsqueeze_1214, unsqueeze_1226, unsqueeze_1238, unsqueeze_1250, unsqueeze_1262, unsqueeze_1274, unsqueeze_1286, unsqueeze_1298, unsqueeze_1310, unsqueeze_1322, unsqueeze_1334, unsqueeze_1346, unsqueeze_1358, unsqueeze_1370, unsqueeze_1382, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('fbnetv3_b', benchmark_compiled_module)
