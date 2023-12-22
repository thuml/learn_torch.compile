
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


# kernel path: /tmp/torchinductor_youkaichao/x6/cx6yd72rj3rdelgl6zdxe2bafg26j5276mi5tk7trzzfuej3mzig.py
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*(r2 // 49)) + (2048*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 49.0
        tmp3 = tmp1 / tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp5 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4a/c4ahnelty27baufwbb73gucmutkl7uikehmihup4mtfcxyozeubw.py
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
    size_hints=[1024, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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


# kernel path: /tmp/torchinductor_youkaichao/2p/c2pdljscu7czzzkv4lizyag6innhu3c2t56mh4szbwsmkpu7bqem.py
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
    size_hints=[1024, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gc/cgc6ohngbln3ppgbxk4y7spajuifxengkb6oiuenpggpreapojb7.py
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
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1024
    x2 = (xindex // 50176)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp5 * tmp11
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7t/c7te7c5mctql2qgpxn3cdt2vabjjpa2ninpfadi4gbbvhhujz3t7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 464
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 232
    x1 = (xindex // 232)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (232*r2) + (22736*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (49 + (98*x0) + (22736*(r2 // 49)) + (45472*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (232*r2) + (22736*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/j2/cj2rm7egl67jv46tjfkqncbjzasckw5h5ssddyihwhmhwh42jin6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 232
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (232*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ed/cedciiigjjsajbjvie6pl2iujgraidfszu5l2gz7kkxug3no37zy.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 232
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (232*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/zg/czgrrtwwm76ib2rx3qfe4kuf5edohfxdjrv2oxc2sv4s7mbyrhyv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
    xnumel = 232
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
    tmp0 = tl.load(in_ptr0 + (x2 + (232*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (49 + y0 + (98*x2) + (22736*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x2 + (232*y3)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cx/ccxvtokmo454d3m2p7zu6ytb7kp2hgkxdannzjp2vtgkujhbe734.py
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
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 232
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (11368*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3g/c3gmrmleymvkyycuvplyjogzjgzt54n6prl7pzqqde47bi3xefzt.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 464
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 232
    x1 = (xindex // 232)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (11368*(r2 // 49)) + (22736*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (232*r2) + (22736*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r3/cr3aj3o77dqol3r7466bpssxa3eugczw55pgm6m2xb5mrlkfkp7a.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 45472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 232
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


# kernel path: /tmp/torchinductor_youkaichao/iw/ciwisbbwg2i6ozsxvcjok46lq45sk4hmpko4wco3n56lt6tg7od7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 464
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 232
    x1 = (xindex // 232)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (232*r2) + (22736*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((49*x0) + (11368*(r2 // 49)) + (22736*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (232*r2) + (22736*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bb/cbbgff3zqmha6md6hga2ungebes3x2ua2c2ueosxqpubfonanj2c.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
    xnumel = 232
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
    tmp0 = tl.load(in_ptr0 + (x2 + (232*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (49*x2) + (11368*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (232*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s7/cs75umlwuqh2nwldmn5bjjkuw2kzaibnbpln7j36earjl3zxt7xb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 232
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex
    r1 = rindex % 49
    r2 = (rindex // 49)
    tmp0 = tl.load(in_ptr0 + (x0 + (232*r3)), rmask & xmask).to(tl.int1)
    tmp1 = 1 + (2*x0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 232, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (r1 + (49*((1 + (2*x0)) // 232)) + (98*((1 + (2*x0)) % 232)) + (22736*r2)), rmask & tmp5 & xmask, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 464, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr2 + ((-11319) + r1 + (98*x0) + (11368*r2)), rmask & tmp9 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = 0.0
    tmp17 = tl.where(tmp0, tmp16, tmp15)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/id/cidqvcymbbtldkwm26mww4g73cm5qunr2conj34mvzw4lr53dvwt.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 464
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 232
    x1 = (xindex // 232)
    x3 = xindex
    tmp19 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (232*r2) + (22736*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp18 = tl.load(in_ptr3 + (x0 + (232*r2) + (22736*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 1 + (2*x0)
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 232, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = tl.load(in_ptr1 + ((49*((1 + (2*x0)) // 232)) + (98*((1 + (2*x0)) % 232)) + (22736*(r2 // 49)) + (45472*x1) + (r2 % 49)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tmp1 >= tmp4
        tmp10 = tl.full([1, 1], 464, tl.int64)
        tmp11 = tmp1 < tmp10
        tmp12 = tl.load(in_ptr2 + ((-11319) + (98*x3) + (11368*(r2 // 49)) + (r2 % 49)), rmask & tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp9, tmp12, tmp13)
        tmp15 = tl.where(tmp5, tmp8, tmp14)
        tmp16 = 0.0
        tmp17 = tl.where(tmp0, tmp16, tmp15)
        tmp20 = tmp18 - tmp19
        tmp21 = tmp17 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/id/cid45skr7jfuxxj3tf73n6itmu7seu4dthw4umk4ptbhzel4eh42.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 928
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 232
    y1 = (yindex // 232)
    tmp0 = tl.load(in_ptr0 + (y0 + (232*x2) + (11368*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp18 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 1 + (2*y0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 232, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (49*((1 + (2*y0)) // 232)) + (98*((1 + (2*y0)) % 232)) + (22736*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 464, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr2 + ((-11319) + x2 + (98*y0) + (11368*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = 0.0
    tmp17 = tl.where(tmp0, tmp16, tmp15)
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (y0 + (232*x2) + (11368*y1)), tmp24, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gt/cgt2ypx4wyefyfyvk5mozonz66i5bgebjbxgsim6h3on6i5rvmqg.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 928
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 232
    y1 = (yindex // 232)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (232*x2) + (11368*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp31 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 1 + (2*y0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 232, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.broadcast_to((2*((1 + (2*y0)) % 232)) + ((1 + (2*y0)) // 232), [XBLOCK, YBLOCK])
    tmp7 = tmp6 >= tmp2
    tmp8 = tmp6 < tmp4
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr1 + (x2 + (49*((((2*((1 + (2*y0)) % 232)) + ((1 + (2*y0)) // 232)) // 232) % 2)) + (98*(((2*((1 + (2*y0)) % 232)) + ((1 + (2*y0)) // 232)) % 232)) + (22736*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tmp6 >= tmp4
    tmp14 = tl.full([1, 1], 464, tl.int64)
    tmp15 = tmp6 < tmp14
    tmp16 = tmp13 & tmp5
    tmp17 = tl.load(in_ptr2 + ((-11368) + x2 + (49*((1 + (2*y0)) // 232)) + (98*((1 + (2*y0)) % 232)) + (11368*y1)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp16, tmp17, tmp18)
    tmp20 = tl.where(tmp8, tmp12, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp5, tmp20, tmp21)
    tmp23 = tmp1 >= tmp4
    tmp24 = tmp1 < tmp14
    tmp25 = tl.load(in_ptr3 + ((-11319) + x2 + (98*y0) + (11368*y1)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp23, tmp25, tmp26)
    tmp28 = tl.where(tmp5, tmp22, tmp27)
    tmp29 = 0.0
    tmp30 = tl.where(tmp0, tmp29, tmp28)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp36 = tmp34 * tmp35
    tmp37 = tmp30 * tmp36
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp30, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (232*x2) + (11368*y1)), tmp37, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/77/c77bmflrlpmgmqfpowwhnubopx2cyoo7zw2lvuclloindteerdmd.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 464
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 232
    x1 = (xindex // 232)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (11368*(r2 // 49)) + (22736*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (232*r2) + (22736*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ec/cecxlf52bb4rm56gklbx26bfucgbrukqcsv4p67ulbsouwpkqw3n.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 90944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49) % 464
    x0 = xindex % 49
    x2 = (xindex // 22736)
    x3 = xindex % 22736
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 232, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (2*(x1 % 232)) + (x1 // 232)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp7 & tmp4
    tmp9 = (2*(((2*(x1 % 232)) + (x1 // 232)) % 232)) + ((((2*(x1 % 232)) + (x1 // 232)) // 232) % 2)
    tmp10 = tmp9 >= tmp1
    tmp11 = tmp9 < tmp3
    tmp12 = tmp11 & tmp8
    tmp13 = tl.load(in_ptr0 + (x0 + (49*((((2*(((2*(x1 % 232)) + (x1 // 232)) % 232)) + ((((2*(x1 % 232)) + (x1 // 232)) // 232) % 2)) // 232) % 2)) + (98*(((2*(((2*(x1 % 232)) + (x1 // 232)) % 232)) + ((((2*(x1 % 232)) + (x1 // 232)) // 232) % 2)) % 232)) + (22736*x2)), tmp12 & xmask, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp9 >= tmp3
    tmp17 = tl.full([1], 464, tl.int64)
    tmp18 = tmp9 < tmp17
    tmp19 = tmp16 & tmp8
    tmp20 = tl.load(in_ptr1 + ((-11368) + x0 + (49*((((2*(x1 % 232)) + (x1 // 232)) // 232) % 2)) + (98*(((2*(x1 % 232)) + (x1 // 232)) % 232)) + (11368*x2)), tmp19 & xmask, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tl.where(tmp11, tmp15, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp8, tmp23, tmp24)
    tmp26 = tmp5 >= tmp3
    tmp27 = tmp5 < tmp17
    tmp28 = tmp26 & tmp4
    tmp29 = tl.load(in_ptr2 + ((-11368) + x0 + (49*(x1 // 232)) + (98*(x1 % 232)) + (11368*x2)), tmp28 & xmask, other=0.0)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp28, tmp29, tmp30)
    tmp32 = tl.where(tmp7, tmp25, tmp31)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp4, tmp32, tmp33)
    tmp35 = tmp0 >= tmp3
    tmp36 = tmp0 < tmp17
    tmp37 = tl.load(in_ptr3 + ((-11368) + x3 + (11368*x2)), tmp35 & xmask, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp35, tmp37, tmp38)
    tmp40 = tl.where(tmp4, tmp34, tmp39)
    tl.store(out_ptr0 + (x4), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2i/c2ixbhen3oy5sul3bay7csz74muoninhssbgcroito6rqraprvot.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1624
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
        tmp0 = tl.load(in_ptr0 + (x1 + (232*r2) + (25984*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (45472*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ki/cki3tnx36v74je45qyqvr6wp6fiuermfi6hoxrx3faygm5cil24v.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 232
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


# kernel path: /tmp/torchinductor_youkaichao/bt/cbtqbsunukcullkc6ey7hnmynenie3dtdfse2trhysyeqwdqyqx2.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1624
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 232
    x1 = (xindex // 232)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (232*r2) + (25984*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (45472*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (232*r2) + (25984*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/xl/cxlm6vwsn5rhe32h5a7gyryly5k66pxpjscsbfsofuevcnh2yoca.py
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
    size_hints=[256, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 232
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (232*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/yk/cykxk4zbfoxok4brerh5btquuasxn5ryfrez62hgvoqsrhpvszc2.py
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
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 232
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
    tmp0 = tl.load(in_ptr0 + (x2 + (232*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (196*x2) + (45472*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (232*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kb/ckbjzmykhfwhcok2tjjssy7ir7gcxmrx3qxfxoilsemtzhytpswt.py
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
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 464
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 232
    x1 = (xindex // 232)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (232*r2) + (22736*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((98*x0) + (22736*(r2 // 49)) + (45472*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (232*r2) + (22736*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/d7/cd7lblnxrfklzvjrh2eqggflmlpxh2vegtqtoow7x4l25he2e6nz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
    xnumel = 232
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
    tmp0 = tl.load(in_ptr0 + (x2 + (232*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (98*x2) + (22736*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x2 + (232*y3)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ct/cctsgz64xvyhkgw3iaw2f2arw5ly7d3rd2v56xzs665hgudkmfka.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
    xnumel = 116
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex
    r1 = rindex % 196
    r2 = (rindex // 196)
    tmp0 = tl.load(in_ptr0 + (x0 + (116*r3)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (196 + r1 + (392*x0) + (45472*r2)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (196 + r1 + (392*x0) + (45472*r2)), rmask & xmask, other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gs/cgsejs4dlpsodn6cezadk5vgazlmywu4oi34myi4mlsgv26btjqn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 812
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (116*r2) + (12992*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (196 + (14*(((r2 + (112*x0)) // 14) % 14)) + (392*x1) + (45472*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (196 + (14*(((r2 + (112*x0)) // 14) % 14)) + (392*x1) + (45472*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x1 + (116*r2) + (12992*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mm/cmmwlpdd2x3zxj3tyzultzhqgxcilebxbhcuwuhp237n7gt52epu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_29', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 116
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


# kernel path: /tmp/torchinductor_youkaichao/mw/cmw3r3ettqednbonkun2usqeu4r5csvuoku76l32mfyor6ngnem2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 464
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 116
    y1 = (yindex // 116)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (116*x2) + (22736*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (196 + x2 + (392*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (196 + x2 + (392*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp5 * tmp11
    tl.store(out_ptr0 + (y0 + (116*x2) + (22736*y1)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/il/cilkgljjihccjsxisb4xblwe2b4f2r26fhc6hjpcdvaxrydtryge.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 116
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (22736*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2v/c2vzrcvca573tjrvfnj4z7zobq5bafgayeunzejmbam6myz47mun.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 812
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
        tmp0 = tl.load(in_ptr0 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (22736*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (116*r2) + (12992*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tw/ctwuzwdxk6bagkmm7mu4l4hgbhgn6jri24ghlv3eeyugc45fxoih.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_33', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 90944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 116
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


# kernel path: /tmp/torchinductor_youkaichao/vc/cvcgdadhp45bire37kw2lnfruu3tufgdif74xhg3utxkfpyufov3.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 812
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
        tmp0 = tl.load(in_ptr0 + (x1 + (116*r2) + (12992*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (22736*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fo/cfofgvqaik2uzohwi37gcxgsfrlw2irycwviu27vnke7fuibgnl4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 116
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


# kernel path: /tmp/torchinductor_youkaichao/mk/cmkvl3wg5jz6iglvynuvbzph65mmnhjg2vnwzfanmandslqwbjxg.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 812
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 116
    x1 = (xindex // 116)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (116*r2) + (12992*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (22736*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (116*r2) + (12992*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/rb/crbpesg62vkogrlaizwllavpnvr3jmdpw3pxm5tbquskgoqxugsx.py
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
    size_hints=[128, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_37', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 116
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (116*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/hd/chdzmcyvl5veelkc3t3i6grgqoog4ctj24tldbfoouoi6fqqbts6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 116
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
    tmp0 = tl.load(in_ptr0 + (x2 + (116*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (196*x2) + (22736*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (116*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g7/cg75jjhobur46ndwydhnzrqi2foza6c2mt4m4mrwzya3yp5tbo7a.py
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
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_39', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel):
    xnumel = 116
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex
    r1 = rindex % 196
    r2 = (rindex // 196)
    tmp0 = tl.load(in_ptr0 + (x0 + (116*r3)), rmask & xmask).to(tl.int1)
    tmp24 = tl.load(in_ptr4 + (x0 + (116*r3)), rmask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 1 + (2*x0)
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 116, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (r1 + (196*((1 + (2*x0)) // 116)) + (392*((1 + (2*x0)) % 116)) + (45472*r2)), rmask & tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (196*((1 + (2*x0)) // 116)) + (392*((1 + (2*x0)) % 116)) + (45472*r2)), rmask & tmp5 & xmask, other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = tmp1 >= tmp4
    tmp12 = tl.full([1], 232, tl.int64)
    tmp13 = tmp1 < tmp12
    tmp14 = tl.load(in_ptr3 + ((-22540) + r1 + (392*x0) + (22736*r2)), rmask & tmp11 & xmask, other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tl.where(tmp5, tmp10, tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp0, tmp18, tmp17)
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp26 = tmp24 - tmp25
    tmp27 = tmp19 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = tl.math.rsqrt(tmp34)
    tmp36 = tmp31 * tmp35
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp36, xmask)
    tl.store(out_ptr0 + (x0), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3h/c3h7ppbrmnm374sophnyslx7wapugigqojtabhnyeuqebdovq4ui.py
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
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 464
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 116
    y1 = (yindex // 116)
    tmp0 = tl.load(in_ptr0 + (y0 + (116*x2) + (22736*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp20 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 1 + (2*y0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 116, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (196*((1 + (2*y0)) // 116)) + (392*((1 + (2*y0)) % 116)) + (45472*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (x2 + (196*((1 + (2*y0)) // 116)) + (392*((1 + (2*y0)) % 116)) + (45472*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = tmp1 >= tmp4
    tmp12 = tl.full([1, 1], 232, tl.int64)
    tmp13 = tmp1 < tmp12
    tmp14 = tl.load(in_ptr3 + ((-22540) + x2 + (392*y0) + (22736*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tl.where(tmp5, tmp10, tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp0, tmp18, tmp17)
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = tl.math.rsqrt(tmp22)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp19 * tmp25
    tl.store(out_ptr0 + (y0 + (116*x2) + (22736*y1)), tmp26, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nd/cndgrh4qc422vlyf6bxthjiufslemqeaoidf4jxqesv33ejc6ref.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 464
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 116
    y1 = (yindex // 116)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (116*x2) + (22736*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp33 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 1 + (2*y0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 116, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.broadcast_to((2*((1 + (2*y0)) % 116)) + ((1 + (2*y0)) // 116), [XBLOCK, YBLOCK])
    tmp7 = tmp6 >= tmp2
    tmp8 = tmp6 < tmp4
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr1 + (x2 + (196*((((2*((1 + (2*y0)) % 116)) + ((1 + (2*y0)) // 116)) // 116) % 2)) + (392*(((2*((1 + (2*y0)) % 116)) + ((1 + (2*y0)) // 116)) % 116)) + (45472*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr2 + (x2 + (196*((((2*((1 + (2*y0)) % 116)) + ((1 + (2*y0)) // 116)) // 116) % 2)) + (392*(((2*((1 + (2*y0)) % 116)) + ((1 + (2*y0)) // 116)) % 116)) + (45472*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tmp6 >= tmp4
    tmp16 = tl.full([1, 1], 232, tl.int64)
    tmp17 = tmp6 < tmp16
    tmp18 = tmp15 & tmp5
    tmp19 = tl.load(in_ptr3 + ((-22736) + x2 + (196*((1 + (2*y0)) // 116)) + (392*((1 + (2*y0)) % 116)) + (22736*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.where(tmp8, tmp14, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp5, tmp22, tmp23)
    tmp25 = tmp1 >= tmp4
    tmp26 = tmp1 < tmp16
    tmp27 = tl.load(in_ptr4 + ((-22540) + x2 + (392*y0) + (22736*y1)), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp25, tmp27, tmp28)
    tmp30 = tl.where(tmp5, tmp24, tmp29)
    tmp31 = 0.0
    tmp32 = tl.where(tmp0, tmp31, tmp30)
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp38 = tmp36 * tmp37
    tmp39 = tmp32 * tmp38
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp32, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (116*x2) + (22736*y1)), tmp39, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/46/c46tmswa4xcypngjusc5hppeumwdxlqwaswkq2ygfxdwzzunvhar.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 181888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 232
    x0 = xindex % 196
    x2 = (xindex // 45472)
    x3 = xindex % 45472
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 116, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (2*(x1 % 116)) + (x1 // 116)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp7 & tmp4
    tmp9 = (2*(((2*(x1 % 116)) + (x1 // 116)) % 116)) + ((((2*(x1 % 116)) + (x1 // 116)) // 116) % 2)
    tmp10 = tmp9 >= tmp1
    tmp11 = tmp9 < tmp3
    tmp12 = tmp11 & tmp8
    tmp13 = tl.load(in_ptr0 + (x0 + (196*((((2*(((2*(x1 % 116)) + (x1 // 116)) % 116)) + ((((2*(x1 % 116)) + (x1 // 116)) // 116) % 2)) // 116) % 2)) + (392*(((2*(((2*(x1 % 116)) + (x1 // 116)) % 116)) + ((((2*(x1 % 116)) + (x1 // 116)) // 116) % 2)) % 116)) + (45472*x2)), tmp12 & xmask, other=0.0)
    tmp14 = tl.load(in_ptr1 + (x0 + (196*((((2*(((2*(x1 % 116)) + (x1 // 116)) % 116)) + ((((2*(x1 % 116)) + (x1 // 116)) // 116) % 2)) // 116) % 2)) + (392*(((2*(((2*(x1 % 116)) + (x1 // 116)) % 116)) + ((((2*(x1 % 116)) + (x1 // 116)) // 116) % 2)) % 116)) + (45472*x2)), tmp12 & xmask, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp12, tmp15, tmp16)
    tmp18 = tmp9 >= tmp3
    tmp19 = tl.full([1], 232, tl.int64)
    tmp20 = tmp9 < tmp19
    tmp21 = tmp18 & tmp8
    tmp22 = tl.load(in_ptr2 + ((-22736) + x0 + (196*((((2*(x1 % 116)) + (x1 // 116)) // 116) % 2)) + (392*(((2*(x1 % 116)) + (x1 // 116)) % 116)) + (22736*x2)), tmp21 & xmask, other=0.0)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp21, tmp22, tmp23)
    tmp25 = tl.where(tmp11, tmp17, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = tmp5 >= tmp3
    tmp29 = tmp5 < tmp19
    tmp30 = tmp28 & tmp4
    tmp31 = tl.load(in_ptr3 + ((-22736) + x0 + (196*(x1 // 116)) + (392*(x1 % 116)) + (22736*x2)), tmp30 & xmask, other=0.0)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp30, tmp31, tmp32)
    tmp34 = tl.where(tmp7, tmp27, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp4, tmp34, tmp35)
    tmp37 = tmp0 >= tmp3
    tmp38 = tmp0 < tmp19
    tmp39 = tl.load(in_ptr4 + ((-22736) + x3 + (22736*x2)), tmp37 & xmask, other=0.0)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp37, tmp39, tmp40)
    tmp42 = tl.where(tmp4, tmp36, tmp41)
    tl.store(out_ptr0 + (x4), tmp42, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vd/cvdvlfxhshkq5lztkx23n4a3pebltacry5ao4pfp7ielgiok3kms.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 812
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
        tmp0 = tl.load(in_ptr0 + (x1 + (116*r2) + (12992*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (196 + (14*(((r2 + (112*x0)) // 14) % 14)) + (392*x1) + (45472*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l7/cl7jxxycc6hnhlrdm5k2jzu7dmmf5x3ov6xlhyygk77nlz3ruywi.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 812
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 116
    x1 = (xindex // 116)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (116*r2) + (12992*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (196 + (14*(((r2 + (112*x1)) // 14) % 14)) + (392*x0) + (45472*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (116*r2) + (12992*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/5i/c5ignriwy26v3n5dauqq5opfo2gdg7txqafwrnlt6z7qelpwmvru.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 116
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
    tmp0 = tl.load(in_ptr0 + (x2 + (116*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (196 + y0 + (392*x2) + (45472*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x2 + (116*y3)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yg/cygbnzfxyf3ipgifosjkxguaflzb7unwagp2cibfrtnmoq4klqfj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
    xnumel = 116
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex
    r1 = rindex % 196
    r2 = (rindex // 196)
    tmp0 = tl.load(in_ptr0 + (x0 + (116*r3)), rmask & xmask).to(tl.int1)
    tmp1 = 1 + (2*x0)
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 116, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (r1 + (196*((1 + (2*x0)) // 116)) + (392*((1 + (2*x0)) % 116)) + (45472*r2)), rmask & tmp5 & xmask, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1], 232, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr2 + ((-22540) + r1 + (392*x0) + (22736*r2)), rmask & tmp9 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = 0.0
    tmp17 = tl.where(tmp0, tmp16, tmp15)
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tl.store(out_ptr0 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ma/cmahhkunst6j4cnqdgkdccja4xrvoxttvewjezzoiuwj5jh6cmeg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 812
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp19 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (116*r2) + (12992*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp18 = tl.load(in_ptr3 + (x1 + (116*r2) + (12992*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 1 + (2*x1)
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 116, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = tl.load(in_ptr1 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*((1 + (2*x1)) // 116)) + (392*((1 + (2*x1)) % 116)) + (45472*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tmp1 >= tmp4
        tmp10 = tl.full([1, 1], 232, tl.int64)
        tmp11 = tmp1 < tmp10
        tmp12 = tl.load(in_ptr2 + ((-22540) + (14*(((r2 + (112*x0)) // 14) % 14)) + (392*x1) + (22736*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp9, tmp12, tmp13)
        tmp15 = tl.where(tmp5, tmp8, tmp14)
        tmp16 = 0.0
        tmp17 = tl.where(tmp0, tmp16, tmp15)
        tmp20 = tmp18 - tmp19
        tmp21 = tmp17 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/th/cthxdooddlv5pyf3bsgxvnlzclramolfl65g4pxzqxgof7z6bhkq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 464
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 116
    y1 = (yindex // 116)
    tmp0 = tl.load(in_ptr0 + (y0 + (116*x2) + (22736*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp18 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 1 + (2*y0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 116, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (196*((1 + (2*y0)) // 116)) + (392*((1 + (2*y0)) % 116)) + (45472*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 232, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr2 + ((-22540) + x2 + (392*y0) + (22736*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = 0.0
    tmp17 = tl.where(tmp0, tmp16, tmp15)
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (y0 + (116*x2) + (22736*y1)), tmp24, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ye/cyevjhdvrjsazoqxed53ntrzcrxu5him57koiszzisrazgjvczgi.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 464
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 116
    y1 = (yindex // 116)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (116*x2) + (22736*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp31 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 1 + (2*y0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 116, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.broadcast_to((2*((1 + (2*y0)) % 116)) + ((1 + (2*y0)) // 116), [XBLOCK, YBLOCK])
    tmp7 = tmp6 >= tmp2
    tmp8 = tmp6 < tmp4
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr1 + (x2 + (196*((((2*((1 + (2*y0)) % 116)) + ((1 + (2*y0)) // 116)) // 116) % 2)) + (392*(((2*((1 + (2*y0)) % 116)) + ((1 + (2*y0)) // 116)) % 116)) + (45472*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tmp6 >= tmp4
    tmp14 = tl.full([1, 1], 232, tl.int64)
    tmp15 = tmp6 < tmp14
    tmp16 = tmp13 & tmp5
    tmp17 = tl.load(in_ptr2 + ((-22736) + x2 + (196*((1 + (2*y0)) // 116)) + (392*((1 + (2*y0)) % 116)) + (22736*y1)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp16, tmp17, tmp18)
    tmp20 = tl.where(tmp8, tmp12, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp5, tmp20, tmp21)
    tmp23 = tmp1 >= tmp4
    tmp24 = tmp1 < tmp14
    tmp25 = tl.load(in_ptr3 + ((-22540) + x2 + (392*y0) + (22736*y1)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp23, tmp25, tmp26)
    tmp28 = tl.where(tmp5, tmp22, tmp27)
    tmp29 = 0.0
    tmp30 = tl.where(tmp0, tmp29, tmp28)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp36 = tmp34 * tmp35
    tmp37 = tmp30 * tmp36
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp30, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (116*x2) + (22736*y1)), tmp37, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wt/cwt2uagtwverq2az3k2nh2vh44abpk4wy7euyi4t4n2vjsaauves.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 181888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 232
    x0 = xindex % 196
    x2 = (xindex // 45472)
    x3 = xindex % 45472
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 116, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (2*(x1 % 116)) + (x1 // 116)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp7 & tmp4
    tmp9 = (2*(((2*(x1 % 116)) + (x1 // 116)) % 116)) + ((((2*(x1 % 116)) + (x1 // 116)) // 116) % 2)
    tmp10 = tmp9 >= tmp1
    tmp11 = tmp9 < tmp3
    tmp12 = tmp11 & tmp8
    tmp13 = tl.load(in_ptr0 + (x0 + (196*((((2*(((2*(x1 % 116)) + (x1 // 116)) % 116)) + ((((2*(x1 % 116)) + (x1 // 116)) // 116) % 2)) // 116) % 2)) + (392*(((2*(((2*(x1 % 116)) + (x1 // 116)) % 116)) + ((((2*(x1 % 116)) + (x1 // 116)) // 116) % 2)) % 116)) + (45472*x2)), tmp12 & xmask, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp9 >= tmp3
    tmp17 = tl.full([1], 232, tl.int64)
    tmp18 = tmp9 < tmp17
    tmp19 = tmp16 & tmp8
    tmp20 = tl.load(in_ptr1 + ((-22736) + x0 + (196*((((2*(x1 % 116)) + (x1 // 116)) // 116) % 2)) + (392*(((2*(x1 % 116)) + (x1 // 116)) % 116)) + (22736*x2)), tmp19 & xmask, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tl.where(tmp11, tmp15, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp8, tmp23, tmp24)
    tmp26 = tmp5 >= tmp3
    tmp27 = tmp5 < tmp17
    tmp28 = tmp26 & tmp4
    tmp29 = tl.load(in_ptr2 + ((-22736) + x0 + (196*(x1 // 116)) + (392*(x1 % 116)) + (22736*x2)), tmp28 & xmask, other=0.0)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp28, tmp29, tmp30)
    tmp32 = tl.where(tmp7, tmp25, tmp31)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp4, tmp32, tmp33)
    tmp35 = tmp0 >= tmp3
    tmp36 = tmp0 < tmp17
    tmp37 = tl.load(in_ptr3 + ((-22736) + x3 + (22736*x2)), tmp35 & xmask, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp35, tmp37, tmp38)
    tmp40 = tl.where(tmp4, tmp34, tmp39)
    tl.store(out_ptr0 + (x4), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f4/cf4mdihzns7xm4wpkzftamh36jniraizm3w6yq3qjtauc5tguq4f.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2900
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
        tmp3 = tl.load(in_ptr0 + (x1 + (116*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((784*x1) + (90944*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wl/cwlod6qkpb4skxqazthpoiewx7jjvxwy5zvfukrzm7efwdg7eif5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 116
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


# kernel path: /tmp/torchinductor_youkaichao/pe/cpelvpgicyeeubgno3ygcmpxzwqspdo7nq5zh2ojv22snxl6zanv.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2900
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 116)
    x0 = xindex % 116
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (116*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((784*x0) + (90944*(((r2 + (126*x1)) // 784) % 4)) + ((r2 + (126*x1)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (116*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/mq/cmqmwq54g4t6asgm6pbn7sb7qpyu7lb6347nb5j7sz3rbeoxqaih.py
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
    size_hints=[128, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_54', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 116
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (116*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/iu/ciupspynk3iarlytxmmqqdbw3rsgm6sidtysdybljquqed3kbjrf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 116
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
    tmp0 = tl.load(in_ptr0 + (x2 + (116*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (90944*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (116*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fl/cfll2ialjeorplptxn7ror5hch6joo3yqnthespok34zf2vtinwv.py
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
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
    xnumel = 116
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex
    r1 = rindex % 196
    r2 = (rindex // 196)
    tmp0 = tl.load(in_ptr0 + (x0 + (116*r3)), rmask & xmask).to(tl.int1)
    tmp1 = 2*x0
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 116, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (r1 + (196*(x0 // 58)) + (392*((2*x0) % 116)) + (45472*r2)), rmask & tmp5 & xmask, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1], 232, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr2 + ((-22736) + r1 + (392*x0) + (22736*r2)), rmask & tmp9 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = 0.0
    tmp17 = tl.where(tmp0, tmp16, tmp15)
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tl.store(out_ptr0 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gf/cgf5f2w5jbtqrkqyhyldflfk4amtb35ogqg7w5qjlpam22ptxckk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 812
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp19 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (116*r2) + (12992*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp18 = tl.load(in_ptr3 + (x1 + (116*r2) + (12992*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 2*x1
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 116, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = tl.load(in_ptr1 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*(x1 // 58)) + (392*((2*x1) % 116)) + (45472*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tmp1 >= tmp4
        tmp10 = tl.full([1, 1], 232, tl.int64)
        tmp11 = tmp1 < tmp10
        tmp12 = tl.load(in_ptr2 + ((-22736) + (14*(((r2 + (112*x0)) // 14) % 14)) + (392*x1) + (22736*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp9, tmp12, tmp13)
        tmp15 = tl.where(tmp5, tmp8, tmp14)
        tmp16 = 0.0
        tmp17 = tl.where(tmp0, tmp16, tmp15)
        tmp20 = tmp18 - tmp19
        tmp21 = tmp17 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4b/c4bizdhld5ch445qb4wslnoytt4nvgxwx2fkfvfm746pn2j5jxmc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 464
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 116
    y1 = (yindex // 116)
    tmp0 = tl.load(in_ptr0 + (y0 + (116*x2) + (22736*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp18 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 2*y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 116, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (196*(y0 // 58)) + (392*((2*y0) % 116)) + (45472*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 232, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr2 + ((-22736) + x2 + (392*y0) + (22736*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = 0.0
    tmp17 = tl.where(tmp0, tmp16, tmp15)
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (y0 + (116*x2) + (22736*y1)), tmp24, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z7/cz7qanlutoxtburhkqkrbdppqwqvw5uixcipz6gfzrllfuf7pdxp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 58
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (x0 + (58*r3)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (784 + r1 + (1568*x0) + (90944*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (784 + r1 + (1568*x0) + (90944*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vz/cvztd6zfot2j3d3vxkaj2l5vfd5zfsesbbkrniidbla5lkifglhp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1450
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 25
    x1 = (xindex // 25)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x0)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (58*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (784 + (1568*x1) + (90944*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (784 + (1568*x1) + (90944*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = 0.0
        tmp8 = tl.where(tmp3, tmp7, tmp6)
        tmp9 = tl.load(in_ptr3 + (x1 + (58*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr4 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp2, tmp12, tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qe/cqexruv3tfhvrsghbxwtb46l4zmk5htggqp2zorino7aipchduow.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_61', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 58
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


# kernel path: /tmp/torchinductor_youkaichao/fv/cfvo5256um2zy4w5263dcuyv4uylhl3fjemcqfyyrmk3r352dkxf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 232
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 58
    y1 = (yindex // 58)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (58*x2) + (45472*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (784 + x2 + (1568*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (784 + x2 + (1568*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp5 * tmp11
    tl.store(out_ptr0 + (y0 + (58*x2) + (45472*y1)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/om/comnzibxykgljd42qupm2yxnlf4tu7awocdnfmbmdefizokjt3uz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_63 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 58
    rnumel = 3136
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (45472*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gv/cgvnoew2ubinxtqlxoqi2i6z7doitadsch6zfn52ynffbab2vvmi.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1450
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
        tmp3 = tl.load(in_ptr0 + ((784*x1) + (45472*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (58*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/cl/cclebinjy3sh5l4wdkxf7fdhvn7u7dwinlqj4abpx6zckm7275qr.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_65', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 181888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 58
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


# kernel path: /tmp/torchinductor_youkaichao/fj/cfj675pzk7uyszuppxdubwtxw6hwqlqrhbua2kzujwwgvpxulrtb.py
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1450
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
        tmp3 = tl.load(in_ptr0 + (x1 + (58*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((784*x1) + (45472*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yt/cytsbdxms3ygucwq7xwe4ygsltq5bqm7j43dilpseqagc7knjkqi.py
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
    size_hints=[64, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 58
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


# kernel path: /tmp/torchinductor_youkaichao/az/cazrvtml4qeq4wbp72nbgkgblsmzslvrvwig5ts4kylt2hatj35n.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1450
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 58)
    x0 = xindex % 58
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (58*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((784*x0) + (45472*(((r2 + (126*x1)) // 784) % 4)) + ((r2 + (126*x1)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (58*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/d2/cd2srbmiatun73zrcgkwie7sfjlewo6inqrjccrqrmsugxqq4emg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_69', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 58
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (58*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/br/cbrn2arpadlxzzcz3f3w5lgo5e4qs3errvi4b3nwyzgu4gbtogdz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 58
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
    tmp0 = tl.load(in_ptr0 + (x2 + (58*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (45472*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (58*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ax/caxib3nbokemkvwwymnbxpqj6usm6ejggxilpjfpxtnci7hv6tli.py
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
    size_hints=[64, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_71', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 58
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp24 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (x0 + (58*r3)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp23 = tl.load(in_ptr4 + (x0 + (58*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 1 + (2*x0)
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 58, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = tl.load(in_ptr1 + (r1 + (784*((1 + (2*x0)) // 58)) + (1568*((1 + (2*x0)) % 58)) + (90944*r2)), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r1 + (784*((1 + (2*x0)) // 58)) + (1568*((1 + (2*x0)) % 58)) + (90944*r2)), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
        tmp10 = tl.where(tmp5, tmp8, tmp9)
        tmp11 = tmp1 >= tmp4
        tmp12 = tl.full([1, 1], 116, tl.int64)
        tmp13 = tmp1 < tmp12
        tmp14 = tl.load(in_ptr3 + ((-44688) + r1 + (1568*x0) + (45472*r2)), rmask & tmp11 & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
        tmp16 = tl.where(tmp11, tmp14, tmp15)
        tmp17 = tl.where(tmp5, tmp10, tmp16)
        tmp18 = 0.0
        tmp19 = tl.where(tmp0, tmp18, tmp17)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
        tmp25 = tmp23 - tmp24
        tmp26 = tmp19 * tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp21, xmask)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tmp30 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = tl.math.rsqrt(tmp32)
    tmp34 = tmp28 * tmp33
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f5/cf5r7mpvhpirwgvvgbofofhe34j3wvpgzaqvl2gdariylkd6r34m.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 232
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 58
    y1 = (yindex // 58)
    tmp0 = tl.load(in_ptr0 + (y0 + (58*x2) + (45472*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp20 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 1 + (2*y0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 58, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (784*((1 + (2*y0)) // 58)) + (1568*((1 + (2*y0)) % 58)) + (90944*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (x2 + (784*((1 + (2*y0)) // 58)) + (1568*((1 + (2*y0)) % 58)) + (90944*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = tmp1 >= tmp4
    tmp12 = tl.full([1, 1], 116, tl.int64)
    tmp13 = tmp1 < tmp12
    tmp14 = tl.load(in_ptr3 + ((-44688) + x2 + (1568*y0) + (45472*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tl.where(tmp5, tmp10, tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp0, tmp18, tmp17)
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = tl.math.rsqrt(tmp22)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp19 * tmp25
    tl.store(out_ptr0 + (y0 + (58*x2) + (45472*y1)), tmp26, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rt/crtgey3mi37ieyjt6nkfsgr42nb5hpq7skngifhaievjex3s6djo.py
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
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 232
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 58
    y1 = (yindex // 58)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (58*x2) + (45472*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp33 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 1 + (2*y0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 58, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.broadcast_to((2*((1 + (2*y0)) % 58)) + ((1 + (2*y0)) // 58), [XBLOCK, YBLOCK])
    tmp7 = tmp6 >= tmp2
    tmp8 = tmp6 < tmp4
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr1 + (x2 + (784*((((2*((1 + (2*y0)) % 58)) + ((1 + (2*y0)) // 58)) // 58) % 2)) + (1568*(((2*((1 + (2*y0)) % 58)) + ((1 + (2*y0)) // 58)) % 58)) + (90944*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr2 + (x2 + (784*((((2*((1 + (2*y0)) % 58)) + ((1 + (2*y0)) // 58)) // 58) % 2)) + (1568*(((2*((1 + (2*y0)) % 58)) + ((1 + (2*y0)) // 58)) % 58)) + (90944*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tmp6 >= tmp4
    tmp16 = tl.full([1, 1], 116, tl.int64)
    tmp17 = tmp6 < tmp16
    tmp18 = tmp15 & tmp5
    tmp19 = tl.load(in_ptr3 + ((-45472) + x2 + (784*((1 + (2*y0)) // 58)) + (1568*((1 + (2*y0)) % 58)) + (45472*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.where(tmp8, tmp14, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp5, tmp22, tmp23)
    tmp25 = tmp1 >= tmp4
    tmp26 = tmp1 < tmp16
    tmp27 = tl.load(in_ptr4 + ((-44688) + x2 + (1568*y0) + (45472*y1)), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp25, tmp27, tmp28)
    tmp30 = tl.where(tmp5, tmp24, tmp29)
    tmp31 = 0.0
    tmp32 = tl.where(tmp0, tmp31, tmp30)
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp38 = tmp36 * tmp37
    tmp39 = tmp32 * tmp38
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp32, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (58*x2) + (45472*y1)), tmp39, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ok/cokae4b4oxbq4wgmxotckcquxyg2a62gkpylsvaak6t5w7glxrm4.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 363776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 116
    x0 = xindex % 784
    x2 = (xindex // 90944)
    x3 = xindex % 90944
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 58, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (2*(x1 % 58)) + (x1 // 58)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp7 & tmp4
    tmp9 = (2*(((2*(x1 % 58)) + (x1 // 58)) % 58)) + ((((2*(x1 % 58)) + (x1 // 58)) // 58) % 2)
    tmp10 = tmp9 >= tmp1
    tmp11 = tmp9 < tmp3
    tmp12 = tmp11 & tmp8
    tmp13 = tl.load(in_ptr0 + (x0 + (784*((((2*(((2*(x1 % 58)) + (x1 // 58)) % 58)) + ((((2*(x1 % 58)) + (x1 // 58)) // 58) % 2)) // 58) % 2)) + (1568*(((2*(((2*(x1 % 58)) + (x1 // 58)) % 58)) + ((((2*(x1 % 58)) + (x1 // 58)) // 58) % 2)) % 58)) + (90944*x2)), tmp12 & xmask, other=0.0)
    tmp14 = tl.load(in_ptr1 + (x0 + (784*((((2*(((2*(x1 % 58)) + (x1 // 58)) % 58)) + ((((2*(x1 % 58)) + (x1 // 58)) // 58) % 2)) // 58) % 2)) + (1568*(((2*(((2*(x1 % 58)) + (x1 // 58)) % 58)) + ((((2*(x1 % 58)) + (x1 // 58)) // 58) % 2)) % 58)) + (90944*x2)), tmp12 & xmask, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp12, tmp15, tmp16)
    tmp18 = tmp9 >= tmp3
    tmp19 = tl.full([1], 116, tl.int64)
    tmp20 = tmp9 < tmp19
    tmp21 = tmp18 & tmp8
    tmp22 = tl.load(in_ptr2 + ((-45472) + x0 + (784*((((2*(x1 % 58)) + (x1 // 58)) // 58) % 2)) + (1568*(((2*(x1 % 58)) + (x1 // 58)) % 58)) + (45472*x2)), tmp21 & xmask, other=0.0)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp21, tmp22, tmp23)
    tmp25 = tl.where(tmp11, tmp17, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = tmp5 >= tmp3
    tmp29 = tmp5 < tmp19
    tmp30 = tmp28 & tmp4
    tmp31 = tl.load(in_ptr3 + ((-45472) + x0 + (784*(x1 // 58)) + (1568*(x1 % 58)) + (45472*x2)), tmp30 & xmask, other=0.0)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp30, tmp31, tmp32)
    tmp34 = tl.where(tmp7, tmp27, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp4, tmp34, tmp35)
    tmp37 = tmp0 >= tmp3
    tmp38 = tmp0 < tmp19
    tmp39 = tl.load(in_ptr4 + ((-45472) + x3 + (45472*x2)), tmp37 & xmask, other=0.0)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp37, tmp39, tmp40)
    tmp42 = tl.where(tmp4, tmp36, tmp41)
    tl.store(out_ptr0 + (x4), tmp42, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ea/ceao5pvhi4gesj6cquft42heyg5cp4koj5rnxi4gfeioj3oka2hv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1450
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
        tmp3 = tl.load(in_ptr0 + (x1 + (58*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (784 + (1568*x1) + (90944*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/tn/ctnyv2pv7urbgizpzvct3xo54dteekl7oia2ljrjbeocgfsd4zsi.py
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1450
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 58)
    x0 = xindex % 58
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (58*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (784 + (1568*x0) + (90944*(((r2 + (126*x1)) // 784) % 4)) + ((r2 + (126*x1)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.load(in_ptr2 + (x0 + (58*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/5r/c5r6ijw3exqktw4mfbbez4ibkg5zzb6itkma3wu64cpsuctfqto5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 58
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
    tmp0 = tl.load(in_ptr0 + (x2 + (58*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (784 + y0 + (1568*x2) + (90944*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x2 + (58*y3)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bz/cbzzoabn7fxv2xmm4jwewd6lweow6tk7n2djkoaycu6eew4hrqv7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5684
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
        tmp0 = tl.load(in_ptr0 + (x1 + (58*r2) + (7424*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x1) + (181888*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/45/c456hy4jrktg7zvrojmpfivgdtidmejzx436aejlq7ybpebqwxub.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_79', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 58
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


# kernel path: /tmp/torchinductor_youkaichao/ip/ciphokvz7tw2ftm35fz4ebt5nfrv77a7wx5opsuq6ljuplz3t5xs.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5684
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 58
    x1 = (xindex // 58)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (58*r2) + (7424*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (181888*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (58*r2) + (7424*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/tz/ctzvtrmblhrpbs3hw45oorb3sinordc2dgvbn7uyuus46gt6unne.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_81', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 58
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
        tmp0 = tl.load(in_ptr0 + (x0 + (58*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/dj/cdjb4nx4kgtch3s5qvi2xwe3oclwu4wi2re6s24yeqn72pad7adf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 58
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
    tmp0 = tl.load(in_ptr0 + (x2 + (58*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (181888*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (58*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kl/cklenj2atqzveonpcl6mw7axdd66sdnoyoh5lfxz7xlv76yxr3km.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_83', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1450
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
        tmp3 = tl.load(in_ptr0 + (x1 + (58*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((1568*x1) + (90944*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/yz/cyz4tfyqjjutkfhvlm7rhe6yqizxesh5tifym5xy22ri3767qe3f.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_84', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1450
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 58)
    x0 = xindex % 58
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (58*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((1568*x0) + (90944*(((r2 + (126*x1)) // 784) % 4)) + ((r2 + (126*x1)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.load(in_ptr2 + (x0 + (58*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ch/cchz4rgdnzbed4cnxxhpjava5dk3zuxi2dmlfp4eyyuw5l5qxfqn.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_85 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_85', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 58
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
    tmp0 = tl.load(in_ptr0 + (x2 + (58*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (1568*x2) + (90944*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 * tmp9
    tl.store(out_ptr0 + (x2 + (58*y3)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4d/c4dbr5x47b73zhi5inwddstipitmb4cifxohx7tj5d24uzhezopg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_86 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_86', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 3136
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (18816*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cn/ccnvlv3azktahjxtt5gfuantbyixdpbs67csogrf7a7uxwskldlz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 600
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
        tmp3 = tl.load(in_ptr0 + ((784*x1) + (18816*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (24*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/un/cunzfxtorueks5wbfvpihfvbzzfhogpghko56csujniwplqg3lud.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_88 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_88', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
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


# kernel path: /tmp/torchinductor_youkaichao/pp/cppyf22jhbzev2nw2tfg2a56gzly77kbyxrdiuemlsw6o2ac6wqh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_89 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_89', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 24
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


# kernel path: /tmp/torchinductor_youkaichao/rj/crjdstqu7uqptjdtbbd3xsfiivdgw4ffzdxblt6d6xvz2aqbomoh.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_90', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i7/ci7ouao4s76jsklatmjpbsic5nltfs6lrrjgnj4ank7d64cq337j.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (24*r2) + (3072*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (24*r2) + (3072*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (24*r2) + (3072*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5hulizcvy3iiicdegtinvylx6cfn3ujfyi6lsbyhvpldxe4ifl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_92 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_92', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24
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
        tmp0 = tl.load(in_ptr0 + (x0 + (24*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w2/cw2wkbte7frrkajx6bnmfnkfhzgv7w3ptvx32tjyud6bva6nksvs.py
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
    size_hints=[32, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_93', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24
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
        tmp0 = tl.load(in_ptr0 + (x0 + (24*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ey/cey2s4xxz3gn7zdo5g3zsxd5fwgzdtak3impsdglpszbpea3efxo.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_94 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_94', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x2), tmp11, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, primals_316, primals_318, primals_319, primals_321, primals_322, primals_324, primals_325, primals_327, primals_328, primals_330, primals_331, primals_333, primals_334, primals_336, primals_337, primals_339, convolution, relu, getitem, getitem_1, convolution_1, add_3, convolution_2, convolution_3, relu_2, convolution_4, add_9, convolution_5, getitem_3, convolution_6, relu_4, convolution_7, add_15, convolution_8, getitem_5, convolution_9, relu_6, convolution_10, add_21, convolution_11, getitem_7, convolution_12, relu_8, convolution_13, add_27, convolution_14, view_7, convolution_15, add_31, convolution_16, convolution_17, relu_11, convolution_18, add_37, convolution_19, getitem_9, convolution_20, relu_13, convolution_21, add_43, convolution_22, getitem_11, convolution_23, relu_15, convolution_24, add_49, convolution_25, getitem_13, convolution_26, relu_17, convolution_27, add_55, convolution_28, getitem_15, convolution_29, relu_19, convolution_30, add_61, convolution_31, getitem_17, convolution_32, relu_21, convolution_33, add_67, convolution_34, getitem_19, convolution_35, relu_23, convolution_36, add_73, convolution_37, getitem_21, convolution_38, relu_25, convolution_39, add_79, convolution_40, view_23, convolution_41, add_83, convolution_42, convolution_43, relu_28, convolution_44, add_89, convolution_45, getitem_23, convolution_46, relu_30, convolution_47, add_95, convolution_48, getitem_25, convolution_49, relu_32, convolution_50, add_101, convolution_51, getitem_27, convolution_52, relu_34, convolution_53, add_107, convolution_54, view_31, convolution_55, mean, permute_17, le, le_1, le_3, le_5, le_7, le_9, le_10, le_12, le_14, le_16, le_18, le_20, le_22, le_24, le_26, le_27, le_29, le_31, le_33, le_35, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (24, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_2, (24, ), (1, ))
    assert_size_stride(primals_4, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_5, (24, ), (1, ))
    assert_size_stride(primals_7, (58, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_8, (58, ), (1, ))
    assert_size_stride(primals_10, (58, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_11, (58, ), (1, ))
    assert_size_stride(primals_13, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_14, (58, ), (1, ))
    assert_size_stride(primals_16, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_17, (58, ), (1, ))
    assert_size_stride(primals_19, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_20, (58, ), (1, ))
    assert_size_stride(primals_22, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (58, ), (1, ))
    assert_size_stride(primals_25, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_26, (58, ), (1, ))
    assert_size_stride(primals_28, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_29, (58, ), (1, ))
    assert_size_stride(primals_31, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_32, (58, ), (1, ))
    assert_size_stride(primals_34, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_35, (58, ), (1, ))
    assert_size_stride(primals_37, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_38, (58, ), (1, ))
    assert_size_stride(primals_40, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_41, (58, ), (1, ))
    assert_size_stride(primals_43, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_44, (58, ), (1, ))
    assert_size_stride(primals_46, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_47, (116, ), (1, ))
    assert_size_stride(primals_49, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_50, (116, ), (1, ))
    assert_size_stride(primals_52, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_53, (116, ), (1, ))
    assert_size_stride(primals_55, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_56, (116, ), (1, ))
    assert_size_stride(primals_58, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_59, (116, ), (1, ))
    assert_size_stride(primals_61, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_62, (116, ), (1, ))
    assert_size_stride(primals_64, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_65, (116, ), (1, ))
    assert_size_stride(primals_67, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_68, (116, ), (1, ))
    assert_size_stride(primals_70, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_71, (116, ), (1, ))
    assert_size_stride(primals_73, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_74, (116, ), (1, ))
    assert_size_stride(primals_76, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_77, (116, ), (1, ))
    assert_size_stride(primals_79, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_80, (116, ), (1, ))
    assert_size_stride(primals_82, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_83, (116, ), (1, ))
    assert_size_stride(primals_85, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_86, (116, ), (1, ))
    assert_size_stride(primals_88, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_89, (116, ), (1, ))
    assert_size_stride(primals_91, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_92, (116, ), (1, ))
    assert_size_stride(primals_94, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_95, (116, ), (1, ))
    assert_size_stride(primals_97, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_98, (116, ), (1, ))
    assert_size_stride(primals_100, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_101, (116, ), (1, ))
    assert_size_stride(primals_103, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_104, (116, ), (1, ))
    assert_size_stride(primals_106, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_107, (116, ), (1, ))
    assert_size_stride(primals_109, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_110, (116, ), (1, ))
    assert_size_stride(primals_112, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_113, (116, ), (1, ))
    assert_size_stride(primals_115, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_116, (116, ), (1, ))
    assert_size_stride(primals_118, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_119, (116, ), (1, ))
    assert_size_stride(primals_121, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(primals_122, (116, ), (1, ))
    assert_size_stride(primals_124, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_125, (232, ), (1, ))
    assert_size_stride(primals_127, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_128, (232, ), (1, ))
    assert_size_stride(primals_130, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_131, (232, ), (1, ))
    assert_size_stride(primals_133, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_134, (232, ), (1, ))
    assert_size_stride(primals_136, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_137, (232, ), (1, ))
    assert_size_stride(primals_139, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_140, (232, ), (1, ))
    assert_size_stride(primals_142, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_143, (232, ), (1, ))
    assert_size_stride(primals_145, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_146, (232, ), (1, ))
    assert_size_stride(primals_148, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_149, (232, ), (1, ))
    assert_size_stride(primals_151, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_152, (232, ), (1, ))
    assert_size_stride(primals_154, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_155, (232, ), (1, ))
    assert_size_stride(primals_157, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_158, (232, ), (1, ))
    assert_size_stride(primals_160, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_161, (232, ), (1, ))
    assert_size_stride(primals_163, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(primals_164, (232, ), (1, ))
    assert_size_stride(primals_166, (1024, 464, 1, 1), (464, 1, 1, 1))
    assert_size_stride(primals_167, (1024, ), (1, ))
    assert_size_stride(primals_171, (24, ), (1, ))
    assert_size_stride(primals_172, (24, ), (1, ))
    assert_size_stride(primals_174, (24, ), (1, ))
    assert_size_stride(primals_175, (24, ), (1, ))
    assert_size_stride(primals_177, (58, ), (1, ))
    assert_size_stride(primals_178, (58, ), (1, ))
    assert_size_stride(primals_180, (58, ), (1, ))
    assert_size_stride(primals_181, (58, ), (1, ))
    assert_size_stride(primals_183, (58, ), (1, ))
    assert_size_stride(primals_184, (58, ), (1, ))
    assert_size_stride(primals_186, (58, ), (1, ))
    assert_size_stride(primals_187, (58, ), (1, ))
    assert_size_stride(primals_189, (58, ), (1, ))
    assert_size_stride(primals_190, (58, ), (1, ))
    assert_size_stride(primals_192, (58, ), (1, ))
    assert_size_stride(primals_193, (58, ), (1, ))
    assert_size_stride(primals_195, (58, ), (1, ))
    assert_size_stride(primals_196, (58, ), (1, ))
    assert_size_stride(primals_198, (58, ), (1, ))
    assert_size_stride(primals_199, (58, ), (1, ))
    assert_size_stride(primals_201, (58, ), (1, ))
    assert_size_stride(primals_202, (58, ), (1, ))
    assert_size_stride(primals_204, (58, ), (1, ))
    assert_size_stride(primals_205, (58, ), (1, ))
    assert_size_stride(primals_207, (58, ), (1, ))
    assert_size_stride(primals_208, (58, ), (1, ))
    assert_size_stride(primals_210, (58, ), (1, ))
    assert_size_stride(primals_211, (58, ), (1, ))
    assert_size_stride(primals_213, (58, ), (1, ))
    assert_size_stride(primals_214, (58, ), (1, ))
    assert_size_stride(primals_216, (116, ), (1, ))
    assert_size_stride(primals_217, (116, ), (1, ))
    assert_size_stride(primals_219, (116, ), (1, ))
    assert_size_stride(primals_220, (116, ), (1, ))
    assert_size_stride(primals_222, (116, ), (1, ))
    assert_size_stride(primals_223, (116, ), (1, ))
    assert_size_stride(primals_225, (116, ), (1, ))
    assert_size_stride(primals_226, (116, ), (1, ))
    assert_size_stride(primals_228, (116, ), (1, ))
    assert_size_stride(primals_229, (116, ), (1, ))
    assert_size_stride(primals_231, (116, ), (1, ))
    assert_size_stride(primals_232, (116, ), (1, ))
    assert_size_stride(primals_234, (116, ), (1, ))
    assert_size_stride(primals_235, (116, ), (1, ))
    assert_size_stride(primals_237, (116, ), (1, ))
    assert_size_stride(primals_238, (116, ), (1, ))
    assert_size_stride(primals_240, (116, ), (1, ))
    assert_size_stride(primals_241, (116, ), (1, ))
    assert_size_stride(primals_243, (116, ), (1, ))
    assert_size_stride(primals_244, (116, ), (1, ))
    assert_size_stride(primals_246, (116, ), (1, ))
    assert_size_stride(primals_247, (116, ), (1, ))
    assert_size_stride(primals_249, (116, ), (1, ))
    assert_size_stride(primals_250, (116, ), (1, ))
    assert_size_stride(primals_252, (116, ), (1, ))
    assert_size_stride(primals_253, (116, ), (1, ))
    assert_size_stride(primals_255, (116, ), (1, ))
    assert_size_stride(primals_256, (116, ), (1, ))
    assert_size_stride(primals_258, (116, ), (1, ))
    assert_size_stride(primals_259, (116, ), (1, ))
    assert_size_stride(primals_261, (116, ), (1, ))
    assert_size_stride(primals_262, (116, ), (1, ))
    assert_size_stride(primals_264, (116, ), (1, ))
    assert_size_stride(primals_265, (116, ), (1, ))
    assert_size_stride(primals_267, (116, ), (1, ))
    assert_size_stride(primals_268, (116, ), (1, ))
    assert_size_stride(primals_270, (116, ), (1, ))
    assert_size_stride(primals_271, (116, ), (1, ))
    assert_size_stride(primals_273, (116, ), (1, ))
    assert_size_stride(primals_274, (116, ), (1, ))
    assert_size_stride(primals_276, (116, ), (1, ))
    assert_size_stride(primals_277, (116, ), (1, ))
    assert_size_stride(primals_279, (116, ), (1, ))
    assert_size_stride(primals_280, (116, ), (1, ))
    assert_size_stride(primals_282, (116, ), (1, ))
    assert_size_stride(primals_283, (116, ), (1, ))
    assert_size_stride(primals_285, (116, ), (1, ))
    assert_size_stride(primals_286, (116, ), (1, ))
    assert_size_stride(primals_288, (116, ), (1, ))
    assert_size_stride(primals_289, (116, ), (1, ))
    assert_size_stride(primals_291, (116, ), (1, ))
    assert_size_stride(primals_292, (116, ), (1, ))
    assert_size_stride(primals_294, (232, ), (1, ))
    assert_size_stride(primals_295, (232, ), (1, ))
    assert_size_stride(primals_297, (232, ), (1, ))
    assert_size_stride(primals_298, (232, ), (1, ))
    assert_size_stride(primals_300, (232, ), (1, ))
    assert_size_stride(primals_301, (232, ), (1, ))
    assert_size_stride(primals_303, (232, ), (1, ))
    assert_size_stride(primals_304, (232, ), (1, ))
    assert_size_stride(primals_306, (232, ), (1, ))
    assert_size_stride(primals_307, (232, ), (1, ))
    assert_size_stride(primals_309, (232, ), (1, ))
    assert_size_stride(primals_310, (232, ), (1, ))
    assert_size_stride(primals_312, (232, ), (1, ))
    assert_size_stride(primals_313, (232, ), (1, ))
    assert_size_stride(primals_315, (232, ), (1, ))
    assert_size_stride(primals_316, (232, ), (1, ))
    assert_size_stride(primals_318, (232, ), (1, ))
    assert_size_stride(primals_319, (232, ), (1, ))
    assert_size_stride(primals_321, (232, ), (1, ))
    assert_size_stride(primals_322, (232, ), (1, ))
    assert_size_stride(primals_324, (232, ), (1, ))
    assert_size_stride(primals_325, (232, ), (1, ))
    assert_size_stride(primals_327, (232, ), (1, ))
    assert_size_stride(primals_328, (232, ), (1, ))
    assert_size_stride(primals_330, (232, ), (1, ))
    assert_size_stride(primals_331, (232, ), (1, ))
    assert_size_stride(primals_333, (232, ), (1, ))
    assert_size_stride(primals_334, (232, ), (1, ))
    assert_size_stride(primals_336, (1024, ), (1, ))
    assert_size_stride(primals_337, (1024, ), (1, ))
    assert_size_stride(primals_339, (4, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (4, 24, 112, 112), (301056, 1, 2688, 24))
    assert_size_stride(relu, (4, 24, 112, 112), (301056, 1, 2688, 24))
    assert_size_stride(getitem, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(getitem_1, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_1, (4, 24, 28, 28), (18816, 1, 672, 24))
    assert_size_stride(add_3, (4, 24, 28, 28), (18816, 1, 672, 24))
    assert_size_stride(convolution_2, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(convolution_3, (4, 58, 56, 56), (181888, 1, 3248, 58))
    assert_size_stride(relu_2, (4, 58, 56, 56), (181888, 1, 3248, 58))
    assert_size_stride(convolution_4, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(add_9, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(convolution_5, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(getitem_3, (4, 58, 28, 28), (90944, 784, 28, 1))
    assert_size_stride(convolution_6, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(relu_4, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(convolution_7, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(add_15, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(convolution_8, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(getitem_5, (4, 58, 28, 28), (90944, 784, 28, 1))
    assert_size_stride(convolution_9, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(relu_6, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(convolution_10, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(add_21, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(convolution_11, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(getitem_7, (4, 58, 28, 28), (90944, 784, 28, 1))
    assert_size_stride(convolution_12, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(relu_8, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(convolution_13, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(add_27, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(convolution_14, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(view_7, (4, 116, 28, 28), (90944, 1, 3248, 116))
    assert_size_stride(convolution_15, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(add_31, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_16, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_17, (4, 116, 28, 28), (90944, 1, 3248, 116))
    assert_size_stride(relu_11, (4, 116, 28, 28), (90944, 1, 3248, 116))
    assert_size_stride(convolution_18, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(add_37, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_19, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(getitem_9, (4, 116, 14, 14), (45472, 196, 14, 1))
    assert_size_stride(convolution_20, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(relu_13, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_21, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(add_43, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_22, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(getitem_11, (4, 116, 14, 14), (45472, 196, 14, 1))
    assert_size_stride(convolution_23, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(relu_15, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_24, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(add_49, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_25, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(getitem_13, (4, 116, 14, 14), (45472, 196, 14, 1))
    assert_size_stride(convolution_26, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(relu_17, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_27, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(add_55, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_28, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(getitem_15, (4, 116, 14, 14), (45472, 196, 14, 1))
    assert_size_stride(convolution_29, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(relu_19, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_30, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(add_61, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_31, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(getitem_17, (4, 116, 14, 14), (45472, 196, 14, 1))
    assert_size_stride(convolution_32, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(relu_21, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_33, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(add_67, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_34, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(getitem_19, (4, 116, 14, 14), (45472, 196, 14, 1))
    assert_size_stride(convolution_35, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(relu_23, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_36, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(add_73, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_37, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(getitem_21, (4, 116, 14, 14), (45472, 196, 14, 1))
    assert_size_stride(convolution_38, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(relu_25, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_39, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(add_79, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(convolution_40, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(view_23, (4, 232, 14, 14), (45472, 1, 3248, 232))
    assert_size_stride(convolution_41, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(add_83, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(convolution_42, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(convolution_43, (4, 232, 14, 14), (45472, 1, 3248, 232))
    assert_size_stride(relu_28, (4, 232, 14, 14), (45472, 1, 3248, 232))
    assert_size_stride(convolution_44, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(add_89, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(convolution_45, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(getitem_23, (4, 232, 7, 7), (22736, 49, 7, 1))
    assert_size_stride(convolution_46, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(relu_30, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(convolution_47, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(add_95, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(convolution_48, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(getitem_25, (4, 232, 7, 7), (22736, 49, 7, 1))
    assert_size_stride(convolution_49, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(relu_32, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(convolution_50, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(add_101, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(convolution_51, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(getitem_27, (4, 232, 7, 7), (22736, 49, 7, 1))
    assert_size_stride(convolution_52, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(relu_34, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(convolution_53, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(add_107, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(convolution_54, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(view_31, (4, 464, 7, 7), (22736, 1, 3248, 464))
    assert_size_stride(convolution_55, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(mean, (4, 1024), (1024, 1))
    assert_size_stride(permute_17, (1000, 1024), (1024, 1))
    assert_size_stride(le, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(le_1, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(le_3, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(le_5, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(le_7, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(le_9, (4, 232, 7, 7), (11368, 1, 1624, 232))
    assert_size_stride(le_10, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(le_12, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(le_14, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(le_16, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(le_18, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(le_20, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(le_22, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(le_24, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(le_26, (4, 116, 14, 14), (22736, 1, 1624, 116))
    assert_size_stride(le_27, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(le_29, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(le_31, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(le_33, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(le_35, (4, 58, 28, 28), (45472, 1, 1624, 58))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_17, out=buf0)
        del permute_17
        buf1 = empty((1000, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), mean, out=buf1)
        del mean
        buf2 = empty((1000, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_sum_view_0.run(tangents_1, buf2, 1000, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty_strided((1024, 2), (1, 1024), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1024, 2), (1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_div_native_batch_norm_backward_threshold_backward_1.run(le, buf0, convolution_55, primals_336, buf3, buf5, 2048, 98, grid=grid(2048), stream=stream0)
        del convolution_55
        del primals_336
        buf4 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_2.run(buf3, buf4, 1024, 2, grid=grid(1024), stream=stream0)
        del buf3
        buf6 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf7 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_3.run(buf7, buf5, primals_337, 1024, 2, grid=grid(1024), stream=stream0)
        del buf5
        buf8 = empty_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4.run(le, buf0, primals_337, primals_167, buf8, 200704, grid=grid(200704), stream=stream0)
        del buf0
        del le
        del primals_167
        del primals_337
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf9 = aten.convolution_backward(buf8, view_31, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf8
        del primals_166
        del view_31
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        buf12 = empty_strided((232, 2), (1, 232), device='cuda', dtype=torch.float32)
        buf14 = empty_strided((232, 2), (1, 232), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(le_1, buf10, convolution_54, primals_333, buf12, buf14, 464, 98, grid=grid(464), stream=stream0)
        del convolution_54
        del primals_333
        buf13 = empty((232, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(buf12, buf13, 232, 2, grid=grid(232), stream=stream0)
        buf15 = empty((232, ), device='cuda', dtype=torch.float32)
        buf16 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf16, buf14, primals_334, 232, 2, grid=grid(232), stream=stream0)
        buf17 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8.run(le_1, buf10, primals_334, primals_164, buf17, 196, 232, grid=grid(196, 232), stream=stream0)
        del le_1
        del primals_164
        del primals_334
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf18 = aten.convolution_backward(buf17, add_107, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_107
        del buf17
        del primals_163
        buf19 = buf18[0]
        buf20 = buf18[1]
        del buf18
        buf21 = empty((232, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_9.run(buf19, buf21, 232, 196, grid=grid(232), stream=stream0)
        buf22 = buf14; del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_10.run(buf19, convolution_53, primals_330, buf22, 464, 98, grid=grid(464), stream=stream0)
        del convolution_53
        del primals_330
        buf23 = empty((232, ), device='cuda', dtype=torch.float32)
        buf24 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf24, buf22, primals_331, 232, 2, grid=grid(232), stream=stream0)
        buf25 = buf19; del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_11.run(buf25, primals_331, primals_161, 45472, grid=grid(45472), stream=stream0)
        del primals_161
        del primals_331
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf26 = aten.convolution_backward(buf25, relu_34, primals_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False])
        del primals_160
        buf27 = buf26[0]
        buf28 = buf26[1]
        del buf26
        buf29 = buf22; del buf22  # reuse
        buf31 = buf12; del buf12  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_12.run(relu_34, buf27, convolution_52, primals_327, buf29, buf31, 464, 98, grid=grid(464), stream=stream0)
        del convolution_52
        del primals_327
        buf30 = empty((232, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(buf29, buf30, 232, 2, grid=grid(232), stream=stream0)
        buf32 = empty((232, ), device='cuda', dtype=torch.float32)
        buf33 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf33, buf31, primals_328, 232, 2, grid=grid(232), stream=stream0)
        buf34 = reinterpret_tensor(buf25, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13.run(relu_34, buf27, primals_328, primals_158, buf34, 196, 232, grid=grid(196, 232), stream=stream0)
        del buf27
        del primals_158
        del primals_328
        del relu_34
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf35 = aten.convolution_backward(buf34, getitem_27, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_27
        del primals_157
        buf36 = buf35[0]
        buf37 = buf35[1]
        del buf35
        buf38 = empty((232, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_14.run(le_3, buf10, buf36, buf38, 232, 196, grid=grid(232), stream=stream0)
        buf39 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_15.run(le_3, buf10, buf36, convolution_51, primals_324, buf39, 464, 98, grid=grid(464), stream=stream0)
        del convolution_51
        del primals_324
        buf40 = empty((232, ), device='cuda', dtype=torch.float32)
        buf41 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf41, buf39, primals_325, 232, 2, grid=grid(232), stream=stream0)
        buf42 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_16.run(le_3, buf10, buf36, primals_325, primals_155, buf42, 928, 49, grid=grid(928, 49), stream=stream0)
        del le_3
        del primals_155
        del primals_325
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf43 = aten.convolution_backward(buf42, add_101, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_101
        del buf42
        del primals_154
        buf44 = buf43[0]
        buf45 = buf43[1]
        del buf43
        buf46 = empty((232, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_9.run(buf44, buf46, 232, 196, grid=grid(232), stream=stream0)
        buf47 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_10.run(buf44, convolution_50, primals_321, buf47, 464, 98, grid=grid(464), stream=stream0)
        del convolution_50
        del primals_321
        buf48 = empty((232, ), device='cuda', dtype=torch.float32)
        buf49 = buf48; del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf49, buf47, primals_322, 232, 2, grid=grid(232), stream=stream0)
        buf50 = buf44; del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_11.run(buf50, primals_322, primals_152, 45472, grid=grid(45472), stream=stream0)
        del primals_152
        del primals_322
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf51 = aten.convolution_backward(buf50, relu_32, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False])
        del primals_151
        buf52 = buf51[0]
        buf53 = buf51[1]
        del buf51
        buf54 = buf47; del buf47  # reuse
        buf56 = buf29; del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_12.run(relu_32, buf52, convolution_49, primals_318, buf54, buf56, 464, 98, grid=grid(464), stream=stream0)
        del convolution_49
        del primals_318
        buf55 = empty((232, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(buf54, buf55, 232, 2, grid=grid(232), stream=stream0)
        buf57 = empty((232, ), device='cuda', dtype=torch.float32)
        buf58 = buf57; del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf58, buf56, primals_319, 232, 2, grid=grid(232), stream=stream0)
        buf59 = reinterpret_tensor(buf50, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13.run(relu_32, buf52, primals_319, primals_149, buf59, 196, 232, grid=grid(196, 232), stream=stream0)
        del primals_149
        del primals_319
        del relu_32
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf60 = aten.convolution_backward(buf59, getitem_25, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_25
        del primals_148
        buf61 = buf60[0]
        buf62 = buf60[1]
        del buf60
        buf63 = reinterpret_tensor(buf59, (4, 232, 7, 7), (11368, 49, 7, 1), 0); del buf59  # reuse
        buf68 = reinterpret_tensor(buf52, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17.run(le_5, buf10, buf36, buf61, primals_316, primals_146, buf63, buf68, 928, 49, grid=grid(928, 49), stream=stream0)
        del le_5
        del primals_146
        buf64 = empty((232, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_9.run(buf63, buf64, 232, 196, grid=grid(232), stream=stream0)
        buf65 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_18.run(buf63, convolution_48, primals_315, buf65, 464, 98, grid=grid(464), stream=stream0)
        del buf63
        del convolution_48
        del primals_315
        buf66 = empty((232, ), device='cuda', dtype=torch.float32)
        buf67 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf67, buf65, primals_316, 232, 2, grid=grid(232), stream=stream0)
        del primals_316
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf69 = aten.convolution_backward(buf68, add_95, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_95
        del buf68
        del primals_145
        buf70 = buf69[0]
        buf71 = buf69[1]
        del buf69
        buf72 = empty((232, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_9.run(buf70, buf72, 232, 196, grid=grid(232), stream=stream0)
        buf73 = buf65; del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_10.run(buf70, convolution_47, primals_312, buf73, 464, 98, grid=grid(464), stream=stream0)
        del convolution_47
        del primals_312
        buf74 = empty((232, ), device='cuda', dtype=torch.float32)
        buf75 = buf74; del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf75, buf73, primals_313, 232, 2, grid=grid(232), stream=stream0)
        buf76 = buf70; del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_11.run(buf76, primals_313, primals_143, 45472, grid=grid(45472), stream=stream0)
        del primals_143
        del primals_313
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf77 = aten.convolution_backward(buf76, relu_30, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False])
        del primals_142
        buf78 = buf77[0]
        buf79 = buf77[1]
        del buf77
        buf80 = buf73; del buf73  # reuse
        buf82 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_12.run(relu_30, buf78, convolution_46, primals_309, buf80, buf82, 464, 98, grid=grid(464), stream=stream0)
        del convolution_46
        del primals_309
        buf81 = empty((232, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(buf80, buf81, 232, 2, grid=grid(232), stream=stream0)
        buf83 = empty((232, ), device='cuda', dtype=torch.float32)
        buf84 = buf83; del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf84, buf82, primals_310, 232, 2, grid=grid(232), stream=stream0)
        buf85 = reinterpret_tensor(buf76, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13.run(relu_30, buf78, primals_310, primals_140, buf85, 196, 232, grid=grid(196, 232), stream=stream0)
        del buf78
        del primals_140
        del primals_310
        del relu_30
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf86 = aten.convolution_backward(buf85, getitem_23, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf85
        del getitem_23
        del primals_139
        buf87 = buf86[0]
        buf88 = buf86[1]
        del buf86
        buf89 = empty((4, 464, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_19.run(buf10, buf36, buf61, buf87, buf89, 90944, grid=grid(90944), stream=stream0)
        del buf10
        del buf36
        del buf61
        buf90 = buf82; del buf82  # reuse
        buf92 = buf80; del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(le_7, buf89, convolution_45, primals_306, buf90, buf92, 464, 98, grid=grid(464), stream=stream0)
        del convolution_45
        del primals_306
        buf91 = empty((232, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(buf90, buf91, 232, 2, grid=grid(232), stream=stream0)
        buf93 = empty((232, ), device='cuda', dtype=torch.float32)
        buf94 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf94, buf92, primals_307, 232, 2, grid=grid(232), stream=stream0)
        buf95 = reinterpret_tensor(buf87, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8.run(le_7, buf89, primals_307, primals_137, buf95, 196, 232, grid=grid(196, 232), stream=stream0)
        del le_7
        del primals_137
        del primals_307
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf96 = aten.convolution_backward(buf95, add_89, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_89
        del buf95
        del primals_136
        buf97 = buf96[0]
        buf98 = buf96[1]
        del buf96
        buf99 = empty((232, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_9.run(buf97, buf99, 232, 196, grid=grid(232), stream=stream0)
        buf100 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_10.run(buf97, convolution_44, primals_303, buf100, 464, 98, grid=grid(464), stream=stream0)
        del convolution_44
        del primals_303
        buf101 = empty((232, ), device='cuda', dtype=torch.float32)
        buf102 = buf101; del buf101  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf102, buf100, primals_304, 232, 2, grid=grid(232), stream=stream0)
        buf103 = buf97; del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_11.run(buf103, primals_304, primals_134, 45472, grid=grid(45472), stream=stream0)
        del primals_134
        del primals_304
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf104 = aten.convolution_backward(buf103, relu_28, primals_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False])
        del primals_133
        buf105 = buf104[0]
        buf106 = buf104[1]
        del buf104
        buf107 = empty((232, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_20.run(relu_28, buf105, buf107, 1624, 112, grid=grid(1624), stream=stream0)
        buf108 = empty((232, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_21.run(buf107, buf108, 232, 7, grid=grid(232), stream=stream0)
        buf109 = reinterpret_tensor(buf107, (232, 7), (1, 232), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(relu_28, buf105, convolution_43, primals_300, buf109, 1624, 112, grid=grid(1624), stream=stream0)
        del convolution_43
        del primals_300
        buf110 = empty((232, ), device='cuda', dtype=torch.float32)
        buf111 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_23.run(buf111, buf109, primals_301, 232, 7, grid=grid(232), stream=stream0)
        del buf109
        buf112 = empty_strided((4, 232, 14, 14), (45472, 1, 3248, 232), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_24.run(relu_28, buf105, primals_301, primals_131, buf112, 784, 232, grid=grid(784, 232), stream=stream0)
        del buf105
        del primals_131
        del primals_301
        del relu_28
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf113 = aten.convolution_backward(buf112, view_23, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_130
        buf114 = buf113[0]
        buf115 = buf113[1]
        del buf113
        buf116 = buf100; del buf100  # reuse
        buf118 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(le_9, buf89, convolution_42, primals_297, buf116, buf118, 464, 98, grid=grid(464), stream=stream0)
        del convolution_42
        del primals_297
        buf117 = empty((232, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(buf116, buf117, 232, 2, grid=grid(232), stream=stream0)
        del buf116
        buf119 = empty((232, ), device='cuda', dtype=torch.float32)
        buf120 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf120, buf118, primals_298, 232, 2, grid=grid(232), stream=stream0)
        buf121 = reinterpret_tensor(buf103, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26.run(le_9, buf89, primals_298, primals_128, buf121, 196, 232, grid=grid(196, 232), stream=stream0)
        del le_9
        del primals_128
        del primals_298
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf122 = aten.convolution_backward(buf121, add_83, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_83
        del buf121
        del primals_127
        buf123 = buf122[0]
        buf124 = buf122[1]
        del buf122
        buf125 = empty((232, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_9.run(buf123, buf125, 232, 196, grid=grid(232), stream=stream0)
        buf126 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_10.run(buf123, convolution_41, primals_294, buf126, 464, 98, grid=grid(464), stream=stream0)
        del convolution_41
        del primals_294
        buf127 = empty((232, ), device='cuda', dtype=torch.float32)
        buf128 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf128, buf126, primals_295, 232, 2, grid=grid(232), stream=stream0)
        del buf126
        buf129 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_11.run(buf129, primals_295, primals_125, 45472, grid=grid(45472), stream=stream0)
        del primals_125
        del primals_295
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf130 = aten.convolution_backward(buf129, view_23, primals_124, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False])
        del buf129
        del primals_124
        del view_23
        buf131 = buf130[0]
        buf132 = buf130[1]
        del buf130
        buf133 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_27.run(le_10, buf114, buf131, buf133, 116, 784, grid=grid(116), stream=stream0)
        buf134 = empty((116, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(le_10, buf114, buf131, convolution_40, primals_291, buf134, 812, 112, grid=grid(812), stream=stream0)
        del convolution_40
        del primals_291
        buf135 = empty((116, ), device='cuda', dtype=torch.float32)
        buf136 = buf135; del buf135  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_29.run(buf136, buf134, primals_292, 116, 7, grid=grid(116), stream=stream0)
        buf137 = reinterpret_tensor(buf89, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_30.run(le_10, buf114, buf131, primals_292, primals_122, buf137, 464, 196, grid=grid(464, 196), stream=stream0)
        del le_10
        del primals_122
        del primals_292
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf138 = aten.convolution_backward(buf137, add_79, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_79
        del buf137
        del primals_121
        buf139 = buf138[0]
        buf140 = buf138[1]
        del buf138
        buf141 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_31.run(buf139, buf141, 116, 784, grid=grid(116), stream=stream0)
        buf142 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf139, convolution_39, primals_288, buf142, 812, 112, grid=grid(812), stream=stream0)
        del convolution_39
        del primals_288
        buf143 = empty((116, ), device='cuda', dtype=torch.float32)
        buf144 = buf143; del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_29.run(buf144, buf142, primals_289, 116, 7, grid=grid(116), stream=stream0)
        buf145 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_33.run(buf145, primals_289, primals_119, 90944, grid=grid(90944), stream=stream0)
        del primals_119
        del primals_289
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf146 = aten.convolution_backward(buf145, relu_25, primals_118, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False])
        del primals_118
        buf147 = buf146[0]
        buf148 = buf146[1]
        del buf146
        buf149 = buf142; del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_34.run(relu_25, buf147, buf149, 812, 112, grid=grid(812), stream=stream0)
        buf150 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_35.run(buf149, buf150, 116, 7, grid=grid(116), stream=stream0)
        buf151 = reinterpret_tensor(buf149, (116, 7), (1, 116), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_25, buf147, convolution_38, primals_285, buf151, 812, 112, grid=grid(812), stream=stream0)
        del convolution_38
        del primals_285
        buf152 = empty((116, ), device='cuda', dtype=torch.float32)
        buf153 = buf152; del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf153, buf151, primals_286, 116, 7, grid=grid(116), stream=stream0)
        buf154 = reinterpret_tensor(buf145, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(relu_25, buf147, primals_286, primals_116, buf154, 784, 116, grid=grid(784, 116), stream=stream0)
        del buf147
        del primals_116
        del primals_286
        del relu_25
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf155 = aten.convolution_backward(buf154, getitem_21, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_21
        del primals_115
        buf156 = buf155[0]
        buf157 = buf155[1]
        del buf155
        buf158 = empty((116, ), device='cuda', dtype=torch.float32)
        buf159 = empty((116, ), device='cuda', dtype=torch.float32)
        buf160 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_39.run(buf160, le_12, buf114, buf131, buf156, convolution_37, primals_282, primals_283, buf158, 116, 784, grid=grid(116), stream=stream0)
        del convolution_37
        del primals_282
        buf161 = buf154; del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40.run(le_12, buf114, buf131, buf156, primals_283, primals_113, buf161, 464, 196, grid=grid(464, 196), stream=stream0)
        del le_12
        del primals_113
        del primals_283
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf162 = aten.convolution_backward(buf161, add_73, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_73
        del buf161
        del primals_112
        buf163 = buf162[0]
        buf164 = buf162[1]
        del buf162
        buf165 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_31.run(buf163, buf165, 116, 784, grid=grid(116), stream=stream0)
        buf166 = reinterpret_tensor(buf151, (116, 7), (7, 1), 0); del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf163, convolution_36, primals_279, buf166, 812, 112, grid=grid(812), stream=stream0)
        del convolution_36
        del primals_279
        buf167 = empty((116, ), device='cuda', dtype=torch.float32)
        buf168 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_29.run(buf168, buf166, primals_280, 116, 7, grid=grid(116), stream=stream0)
        buf169 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_33.run(buf169, primals_280, primals_110, 90944, grid=grid(90944), stream=stream0)
        del primals_110
        del primals_280
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf170 = aten.convolution_backward(buf169, relu_23, primals_109, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False])
        del primals_109
        buf171 = buf170[0]
        buf172 = buf170[1]
        del buf170
        buf173 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_34.run(relu_23, buf171, buf173, 812, 112, grid=grid(812), stream=stream0)
        buf174 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_35.run(buf173, buf174, 116, 7, grid=grid(116), stream=stream0)
        buf175 = reinterpret_tensor(buf173, (116, 7), (1, 116), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_23, buf171, convolution_35, primals_276, buf175, 812, 112, grid=grid(812), stream=stream0)
        del convolution_35
        del primals_276
        buf176 = empty((116, ), device='cuda', dtype=torch.float32)
        buf177 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf177, buf175, primals_277, 116, 7, grid=grid(116), stream=stream0)
        buf178 = reinterpret_tensor(buf169, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(relu_23, buf171, primals_277, primals_107, buf178, 784, 116, grid=grid(784, 116), stream=stream0)
        del primals_107
        del primals_277
        del relu_23
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf179 = aten.convolution_backward(buf178, getitem_19, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_19
        del primals_106
        buf180 = buf179[0]
        buf181 = buf179[1]
        del buf179
        buf182 = reinterpret_tensor(buf178, (4, 116, 14, 14), (22736, 196, 14, 1), 0); del buf178  # reuse
        buf187 = reinterpret_tensor(buf171, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41.run(le_14, buf114, buf131, buf156, buf180, primals_274, primals_104, buf182, buf187, 464, 196, grid=grid(464, 196), stream=stream0)
        del le_14
        del primals_104
        buf183 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_31.run(buf182, buf183, 116, 784, grid=grid(116), stream=stream0)
        buf184 = reinterpret_tensor(buf175, (116, 7), (7, 1), 0); del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf182, convolution_34, primals_273, buf184, 812, 112, grid=grid(812), stream=stream0)
        del buf182
        del convolution_34
        del primals_273
        buf185 = empty((116, ), device='cuda', dtype=torch.float32)
        buf186 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_29.run(buf186, buf184, primals_274, 116, 7, grid=grid(116), stream=stream0)
        del primals_274
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf188 = aten.convolution_backward(buf187, add_67, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_67
        del buf187
        del primals_103
        buf189 = buf188[0]
        buf190 = buf188[1]
        del buf188
        buf191 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_31.run(buf189, buf191, 116, 784, grid=grid(116), stream=stream0)
        buf192 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf189, convolution_33, primals_270, buf192, 812, 112, grid=grid(812), stream=stream0)
        del convolution_33
        del primals_270
        buf193 = empty((116, ), device='cuda', dtype=torch.float32)
        buf194 = buf193; del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_29.run(buf194, buf192, primals_271, 116, 7, grid=grid(116), stream=stream0)
        buf195 = buf189; del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_33.run(buf195, primals_271, primals_101, 90944, grid=grid(90944), stream=stream0)
        del primals_101
        del primals_271
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf196 = aten.convolution_backward(buf195, relu_21, primals_100, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False])
        del primals_100
        buf197 = buf196[0]
        buf198 = buf196[1]
        del buf196
        buf199 = buf192; del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_34.run(relu_21, buf197, buf199, 812, 112, grid=grid(812), stream=stream0)
        buf200 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_35.run(buf199, buf200, 116, 7, grid=grid(116), stream=stream0)
        buf201 = reinterpret_tensor(buf199, (116, 7), (1, 116), 0); del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_21, buf197, convolution_32, primals_267, buf201, 812, 112, grid=grid(812), stream=stream0)
        del convolution_32
        del primals_267
        buf202 = empty((116, ), device='cuda', dtype=torch.float32)
        buf203 = buf202; del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf203, buf201, primals_268, 116, 7, grid=grid(116), stream=stream0)
        buf204 = reinterpret_tensor(buf195, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(relu_21, buf197, primals_268, primals_98, buf204, 784, 116, grid=grid(784, 116), stream=stream0)
        del buf197
        del primals_268
        del primals_98
        del relu_21
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf205 = aten.convolution_backward(buf204, getitem_17, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf204
        del getitem_17
        del primals_97
        buf206 = buf205[0]
        buf207 = buf205[1]
        del buf205
        buf208 = reinterpret_tensor(buf112, (4, 232, 14, 14), (45472, 196, 14, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_42.run(buf114, buf131, buf156, buf180, buf206, buf208, 181888, grid=grid(181888), stream=stream0)
        del buf114
        del buf156
        del buf180
        buf209 = reinterpret_tensor(buf201, (116, 7), (7, 1), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(le_16, buf208, buf209, 812, 112, grid=grid(812), stream=stream0)
        buf210 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_35.run(buf209, buf210, 116, 7, grid=grid(116), stream=stream0)
        buf211 = reinterpret_tensor(buf209, (116, 7), (1, 116), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(le_16, buf208, convolution_31, primals_264, buf211, 812, 112, grid=grid(812), stream=stream0)
        del convolution_31
        del primals_264
        buf212 = empty((116, ), device='cuda', dtype=torch.float32)
        buf213 = buf212; del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf213, buf211, primals_265, 116, 7, grid=grid(116), stream=stream0)
        buf214 = reinterpret_tensor(buf206, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45.run(le_16, buf208, primals_265, primals_95, buf214, 784, 116, grid=grid(784, 116), stream=stream0)
        del le_16
        del primals_265
        del primals_95
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf215 = aten.convolution_backward(buf214, add_61, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_61
        del buf214
        del primals_94
        buf216 = buf215[0]
        buf217 = buf215[1]
        del buf215
        buf218 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_31.run(buf216, buf218, 116, 784, grid=grid(116), stream=stream0)
        buf219 = reinterpret_tensor(buf211, (116, 7), (7, 1), 0); del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf216, convolution_30, primals_261, buf219, 812, 112, grid=grid(812), stream=stream0)
        del convolution_30
        del primals_261
        buf220 = empty((116, ), device='cuda', dtype=torch.float32)
        buf221 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_29.run(buf221, buf219, primals_262, 116, 7, grid=grid(116), stream=stream0)
        buf222 = buf216; del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_33.run(buf222, primals_262, primals_92, 90944, grid=grid(90944), stream=stream0)
        del primals_262
        del primals_92
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf223 = aten.convolution_backward(buf222, relu_19, primals_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False])
        del primals_91
        buf224 = buf223[0]
        buf225 = buf223[1]
        del buf223
        buf226 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_34.run(relu_19, buf224, buf226, 812, 112, grid=grid(812), stream=stream0)
        buf227 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_35.run(buf226, buf227, 116, 7, grid=grid(116), stream=stream0)
        buf228 = reinterpret_tensor(buf226, (116, 7), (1, 116), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_19, buf224, convolution_29, primals_258, buf228, 812, 112, grid=grid(812), stream=stream0)
        del convolution_29
        del primals_258
        buf229 = empty((116, ), device='cuda', dtype=torch.float32)
        buf230 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf230, buf228, primals_259, 116, 7, grid=grid(116), stream=stream0)
        buf231 = reinterpret_tensor(buf222, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(relu_19, buf224, primals_259, primals_89, buf231, 784, 116, grid=grid(784, 116), stream=stream0)
        del buf224
        del primals_259
        del primals_89
        del relu_19
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf232 = aten.convolution_backward(buf231, getitem_15, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_15
        del primals_88
        buf233 = buf232[0]
        buf234 = buf232[1]
        del buf232
        buf235 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_46.run(le_18, buf208, buf233, buf235, 116, 784, grid=grid(116), stream=stream0)
        buf236 = reinterpret_tensor(buf228, (116, 7), (7, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_47.run(le_18, buf208, buf233, convolution_28, primals_255, buf236, 812, 112, grid=grid(812), stream=stream0)
        del convolution_28
        del primals_255
        buf237 = empty((116, ), device='cuda', dtype=torch.float32)
        buf238 = buf237; del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_29.run(buf238, buf236, primals_256, 116, 7, grid=grid(116), stream=stream0)
        buf239 = buf231; del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_48.run(le_18, buf208, buf233, primals_256, primals_86, buf239, 464, 196, grid=grid(464, 196), stream=stream0)
        del le_18
        del primals_256
        del primals_86
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf240 = aten.convolution_backward(buf239, add_55, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_55
        del buf239
        del primals_85
        buf241 = buf240[0]
        buf242 = buf240[1]
        del buf240
        buf243 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_31.run(buf241, buf243, 116, 784, grid=grid(116), stream=stream0)
        buf244 = buf236; del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf241, convolution_27, primals_252, buf244, 812, 112, grid=grid(812), stream=stream0)
        del convolution_27
        del primals_252
        buf245 = empty((116, ), device='cuda', dtype=torch.float32)
        buf246 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_29.run(buf246, buf244, primals_253, 116, 7, grid=grid(116), stream=stream0)
        buf247 = buf241; del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_33.run(buf247, primals_253, primals_83, 90944, grid=grid(90944), stream=stream0)
        del primals_253
        del primals_83
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf248 = aten.convolution_backward(buf247, relu_17, primals_82, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False])
        del primals_82
        buf249 = buf248[0]
        buf250 = buf248[1]
        del buf248
        buf251 = buf244; del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_34.run(relu_17, buf249, buf251, 812, 112, grid=grid(812), stream=stream0)
        buf252 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_35.run(buf251, buf252, 116, 7, grid=grid(116), stream=stream0)
        buf253 = reinterpret_tensor(buf251, (116, 7), (1, 116), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_17, buf249, convolution_26, primals_249, buf253, 812, 112, grid=grid(812), stream=stream0)
        del convolution_26
        del primals_249
        buf254 = empty((116, ), device='cuda', dtype=torch.float32)
        buf255 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf255, buf253, primals_250, 116, 7, grid=grid(116), stream=stream0)
        buf256 = reinterpret_tensor(buf247, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(relu_17, buf249, primals_250, primals_80, buf256, 784, 116, grid=grid(784, 116), stream=stream0)
        del primals_250
        del primals_80
        del relu_17
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf257 = aten.convolution_backward(buf256, getitem_13, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_13
        del primals_79
        buf258 = buf257[0]
        buf259 = buf257[1]
        del buf257
        buf260 = reinterpret_tensor(buf256, (4, 116, 14, 14), (22736, 196, 14, 1), 0); del buf256  # reuse
        buf265 = reinterpret_tensor(buf249, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49.run(le_20, buf208, buf233, buf258, primals_247, primals_77, buf260, buf265, 464, 196, grid=grid(464, 196), stream=stream0)
        del le_20
        del primals_77
        buf261 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_31.run(buf260, buf261, 116, 784, grid=grid(116), stream=stream0)
        buf262 = reinterpret_tensor(buf253, (116, 7), (7, 1), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf260, convolution_25, primals_246, buf262, 812, 112, grid=grid(812), stream=stream0)
        del buf260
        del convolution_25
        del primals_246
        buf263 = empty((116, ), device='cuda', dtype=torch.float32)
        buf264 = buf263; del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_29.run(buf264, buf262, primals_247, 116, 7, grid=grid(116), stream=stream0)
        del primals_247
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf266 = aten.convolution_backward(buf265, add_49, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_49
        del buf265
        del primals_76
        buf267 = buf266[0]
        buf268 = buf266[1]
        del buf266
        buf269 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_31.run(buf267, buf269, 116, 784, grid=grid(116), stream=stream0)
        buf270 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf267, convolution_24, primals_243, buf270, 812, 112, grid=grid(812), stream=stream0)
        del convolution_24
        del primals_243
        buf271 = empty((116, ), device='cuda', dtype=torch.float32)
        buf272 = buf271; del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_29.run(buf272, buf270, primals_244, 116, 7, grid=grid(116), stream=stream0)
        buf273 = buf267; del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_33.run(buf273, primals_244, primals_74, 90944, grid=grid(90944), stream=stream0)
        del primals_244
        del primals_74
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf274 = aten.convolution_backward(buf273, relu_15, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False])
        del primals_73
        buf275 = buf274[0]
        buf276 = buf274[1]
        del buf274
        buf277 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_34.run(relu_15, buf275, buf277, 812, 112, grid=grid(812), stream=stream0)
        buf278 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_35.run(buf277, buf278, 116, 7, grid=grid(116), stream=stream0)
        buf279 = reinterpret_tensor(buf277, (116, 7), (1, 116), 0); del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_15, buf275, convolution_23, primals_240, buf279, 812, 112, grid=grid(812), stream=stream0)
        del convolution_23
        del primals_240
        buf280 = empty((116, ), device='cuda', dtype=torch.float32)
        buf281 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf281, buf279, primals_241, 116, 7, grid=grid(116), stream=stream0)
        buf282 = reinterpret_tensor(buf273, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(relu_15, buf275, primals_241, primals_71, buf282, 784, 116, grid=grid(784, 116), stream=stream0)
        del buf275
        del primals_241
        del primals_71
        del relu_15
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf283 = aten.convolution_backward(buf282, getitem_11, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf282
        del getitem_11
        del primals_70
        buf284 = buf283[0]
        buf285 = buf283[1]
        del buf283
        buf286 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_50.run(buf208, buf233, buf258, buf284, buf286, 181888, grid=grid(181888), stream=stream0)
        del buf208
        del buf233
        del buf258
        buf287 = reinterpret_tensor(buf279, (116, 7), (7, 1), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_43.run(le_22, buf286, buf287, 812, 112, grid=grid(812), stream=stream0)
        buf288 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_35.run(buf287, buf288, 116, 7, grid=grid(116), stream=stream0)
        buf289 = reinterpret_tensor(buf287, (116, 7), (1, 116), 0); del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_44.run(le_22, buf286, convolution_22, primals_237, buf289, 812, 112, grid=grid(812), stream=stream0)
        del convolution_22
        del primals_237
        buf290 = empty((116, ), device='cuda', dtype=torch.float32)
        buf291 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf291, buf289, primals_238, 116, 7, grid=grid(116), stream=stream0)
        buf292 = reinterpret_tensor(buf284, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45.run(le_22, buf286, primals_238, primals_68, buf292, 784, 116, grid=grid(784, 116), stream=stream0)
        del le_22
        del primals_238
        del primals_68
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf293 = aten.convolution_backward(buf292, add_43, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_43
        del buf292
        del primals_67
        buf294 = buf293[0]
        buf295 = buf293[1]
        del buf293
        buf296 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_31.run(buf294, buf296, 116, 784, grid=grid(116), stream=stream0)
        buf297 = reinterpret_tensor(buf289, (116, 7), (7, 1), 0); del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf294, convolution_21, primals_234, buf297, 812, 112, grid=grid(812), stream=stream0)
        del convolution_21
        del primals_234
        buf298 = empty((116, ), device='cuda', dtype=torch.float32)
        buf299 = buf298; del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_29.run(buf299, buf297, primals_235, 116, 7, grid=grid(116), stream=stream0)
        buf300 = buf294; del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_33.run(buf300, primals_235, primals_65, 90944, grid=grid(90944), stream=stream0)
        del primals_235
        del primals_65
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf301 = aten.convolution_backward(buf300, relu_13, primals_64, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False])
        del primals_64
        buf302 = buf301[0]
        buf303 = buf301[1]
        del buf301
        buf304 = buf297; del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_34.run(relu_13, buf302, buf304, 812, 112, grid=grid(812), stream=stream0)
        buf305 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_35.run(buf304, buf305, 116, 7, grid=grid(116), stream=stream0)
        buf306 = reinterpret_tensor(buf304, (116, 7), (1, 116), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(relu_13, buf302, convolution_20, primals_231, buf306, 812, 112, grid=grid(812), stream=stream0)
        del convolution_20
        del primals_231
        buf307 = empty((116, ), device='cuda', dtype=torch.float32)
        buf308 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_37.run(buf308, buf306, primals_232, 116, 7, grid=grid(116), stream=stream0)
        buf309 = reinterpret_tensor(buf300, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(relu_13, buf302, primals_232, primals_62, buf309, 784, 116, grid=grid(784, 116), stream=stream0)
        del buf302
        del primals_232
        del primals_62
        del relu_13
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf310 = aten.convolution_backward(buf309, getitem_9, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_9
        del primals_61
        buf311 = buf310[0]
        buf312 = buf310[1]
        del buf310
        buf313 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_46.run(le_24, buf286, buf311, buf313, 116, 784, grid=grid(116), stream=stream0)
        buf314 = reinterpret_tensor(buf306, (116, 7), (7, 1), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_47.run(le_24, buf286, buf311, convolution_19, primals_228, buf314, 812, 112, grid=grid(812), stream=stream0)
        del convolution_19
        del primals_228
        buf315 = empty((116, ), device='cuda', dtype=torch.float32)
        buf316 = buf315; del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_29.run(buf316, buf314, primals_229, 116, 7, grid=grid(116), stream=stream0)
        buf317 = buf309; del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_48.run(le_24, buf286, buf311, primals_229, primals_59, buf317, 464, 196, grid=grid(464, 196), stream=stream0)
        del le_24
        del primals_229
        del primals_59
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf318 = aten.convolution_backward(buf317, add_37, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_37
        del buf317
        del primals_58
        buf319 = buf318[0]
        buf320 = buf318[1]
        del buf318
        buf321 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_31.run(buf319, buf321, 116, 784, grid=grid(116), stream=stream0)
        buf322 = buf314; del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf319, convolution_18, primals_225, buf322, 812, 112, grid=grid(812), stream=stream0)
        del convolution_18
        del primals_225
        buf323 = empty((116, ), device='cuda', dtype=torch.float32)
        buf324 = buf323; del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_29.run(buf324, buf322, primals_226, 116, 7, grid=grid(116), stream=stream0)
        buf325 = buf319; del buf319  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_33.run(buf325, primals_226, primals_56, 90944, grid=grid(90944), stream=stream0)
        del primals_226
        del primals_56
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf326 = aten.convolution_backward(buf325, relu_11, primals_55, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False])
        del primals_55
        buf327 = buf326[0]
        buf328 = buf326[1]
        del buf326
        buf329 = empty((116, 25), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_51.run(relu_11, buf327, buf329, 2900, 126, grid=grid(2900), stream=stream0)
        buf330 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_52.run(buf329, buf330, 116, 25, grid=grid(116), stream=stream0)
        buf331 = reinterpret_tensor(buf329, (116, 25), (1, 116), 0); del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_53.run(relu_11, buf327, convolution_17, primals_222, buf331, 2900, 126, grid=grid(2900), stream=stream0)
        del convolution_17
        del primals_222
        buf332 = empty((116, ), device='cuda', dtype=torch.float32)
        buf333 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_54.run(buf333, buf331, primals_223, 116, 25, grid=grid(116), stream=stream0)
        del buf331
        buf334 = empty_strided((4, 116, 28, 28), (90944, 1, 3248, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_55.run(relu_11, buf327, primals_223, primals_53, buf334, 3136, 116, grid=grid(3136, 116), stream=stream0)
        del buf327
        del primals_223
        del primals_53
        del relu_11
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf335 = aten.convolution_backward(buf334, view_7, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_52
        buf336 = buf335[0]
        buf337 = buf335[1]
        del buf335
        buf338 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_56.run(le_26, buf286, buf311, buf338, 116, 784, grid=grid(116), stream=stream0)
        buf339 = buf322; del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_57.run(le_26, buf286, buf311, convolution_16, primals_219, buf339, 812, 112, grid=grid(812), stream=stream0)
        del convolution_16
        del primals_219
        buf340 = empty((116, ), device='cuda', dtype=torch.float32)
        buf341 = buf340; del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_29.run(buf341, buf339, primals_220, 116, 7, grid=grid(116), stream=stream0)
        buf342 = reinterpret_tensor(buf325, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_58.run(le_26, buf286, buf311, primals_220, primals_50, buf342, 464, 196, grid=grid(464, 196), stream=stream0)
        del buf311
        del le_26
        del primals_220
        del primals_50
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf343 = aten.convolution_backward(buf342, add_31, primals_49, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_31
        del buf342
        del primals_49
        buf344 = buf343[0]
        buf345 = buf343[1]
        del buf343
        buf346 = empty((116, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_31.run(buf344, buf346, 116, 784, grid=grid(116), stream=stream0)
        buf347 = buf339; del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf344, convolution_15, primals_216, buf347, 812, 112, grid=grid(812), stream=stream0)
        del convolution_15
        del primals_216
        buf348 = empty((116, ), device='cuda', dtype=torch.float32)
        buf349 = buf348; del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_29.run(buf349, buf347, primals_217, 116, 7, grid=grid(116), stream=stream0)
        del buf347
        buf350 = buf344; del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_33.run(buf350, primals_217, primals_47, 90944, grid=grid(90944), stream=stream0)
        del primals_217
        del primals_47
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf351 = aten.convolution_backward(buf350, view_7, primals_46, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False])
        del buf350
        del primals_46
        del view_7
        buf352 = buf351[0]
        buf353 = buf351[1]
        del buf351
        buf354 = empty((58, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_59.run(le_27, buf336, buf352, buf354, 58, 3136, grid=grid(58), stream=stream0)
        buf355 = empty((58, 25), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_60.run(le_27, buf336, buf352, convolution_14, primals_213, buf355, 1450, 126, grid=grid(1450), stream=stream0)
        del convolution_14
        del primals_213
        buf356 = empty((58, ), device='cuda', dtype=torch.float32)
        buf357 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_61.run(buf357, buf355, primals_214, 58, 25, grid=grid(58), stream=stream0)
        buf358 = reinterpret_tensor(buf286, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_62.run(le_27, buf336, buf352, primals_214, primals_44, buf358, 232, 784, grid=grid(232, 784), stream=stream0)
        del le_27
        del primals_214
        del primals_44
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf359 = aten.convolution_backward(buf358, add_27, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_27
        del buf358
        del primals_43
        buf360 = buf359[0]
        buf361 = buf359[1]
        del buf359
        buf362 = empty((58, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_63.run(buf360, buf362, 58, 3136, grid=grid(58), stream=stream0)
        buf363 = buf355; del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_64.run(buf360, convolution_13, primals_210, buf363, 1450, 126, grid=grid(1450), stream=stream0)
        del convolution_13
        del primals_210
        buf364 = empty((58, ), device='cuda', dtype=torch.float32)
        buf365 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_61.run(buf365, buf363, primals_211, 58, 25, grid=grid(58), stream=stream0)
        buf366 = buf360; del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_65.run(buf366, primals_211, primals_41, 181888, grid=grid(181888), stream=stream0)
        del primals_211
        del primals_41
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf367 = aten.convolution_backward(buf366, relu_8, primals_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False])
        del primals_40
        buf368 = buf367[0]
        buf369 = buf367[1]
        del buf367
        buf370 = buf363; del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(relu_8, buf368, buf370, 1450, 126, grid=grid(1450), stream=stream0)
        buf371 = empty((58, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf370, buf371, 58, 25, grid=grid(58), stream=stream0)
        buf372 = reinterpret_tensor(buf370, (58, 25), (1, 58), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_68.run(relu_8, buf368, convolution_12, primals_207, buf372, 1450, 126, grid=grid(1450), stream=stream0)
        del convolution_12
        del primals_207
        buf373 = empty((58, ), device='cuda', dtype=torch.float32)
        buf374 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_69.run(buf374, buf372, primals_208, 58, 25, grid=grid(58), stream=stream0)
        buf375 = reinterpret_tensor(buf366, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70.run(relu_8, buf368, primals_208, primals_38, buf375, 3136, 58, grid=grid(3136, 58), stream=stream0)
        del buf368
        del primals_208
        del primals_38
        del relu_8
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf376 = aten.convolution_backward(buf375, getitem_7, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_7
        del primals_37
        buf377 = buf376[0]
        buf378 = buf376[1]
        del buf376
        buf379 = empty((58, ), device='cuda', dtype=torch.float32)
        buf380 = empty((58, ), device='cuda', dtype=torch.float32)
        buf381 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(buf381, le_29, buf336, buf352, buf377, convolution_11, primals_204, primals_205, buf379, 58, 3136, grid=grid(58), stream=stream0)
        del convolution_11
        del primals_204
        buf382 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_72.run(le_29, buf336, buf352, buf377, primals_205, primals_35, buf382, 232, 784, grid=grid(232, 784), stream=stream0)
        del le_29
        del primals_205
        del primals_35
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf383 = aten.convolution_backward(buf382, add_21, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_21
        del buf382
        del primals_34
        buf384 = buf383[0]
        buf385 = buf383[1]
        del buf383
        buf386 = empty((58, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_63.run(buf384, buf386, 58, 3136, grid=grid(58), stream=stream0)
        buf387 = reinterpret_tensor(buf372, (58, 25), (25, 1), 0); del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_64.run(buf384, convolution_10, primals_201, buf387, 1450, 126, grid=grid(1450), stream=stream0)
        del convolution_10
        del primals_201
        buf388 = empty((58, ), device='cuda', dtype=torch.float32)
        buf389 = buf388; del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_61.run(buf389, buf387, primals_202, 58, 25, grid=grid(58), stream=stream0)
        buf390 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_65.run(buf390, primals_202, primals_32, 181888, grid=grid(181888), stream=stream0)
        del primals_202
        del primals_32
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf391 = aten.convolution_backward(buf390, relu_6, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False])
        del primals_31
        buf392 = buf391[0]
        buf393 = buf391[1]
        del buf391
        buf394 = buf387; del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(relu_6, buf392, buf394, 1450, 126, grid=grid(1450), stream=stream0)
        buf395 = empty((58, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf394, buf395, 58, 25, grid=grid(58), stream=stream0)
        buf396 = reinterpret_tensor(buf394, (58, 25), (1, 58), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_68.run(relu_6, buf392, convolution_9, primals_198, buf396, 1450, 126, grid=grid(1450), stream=stream0)
        del convolution_9
        del primals_198
        buf397 = empty((58, ), device='cuda', dtype=torch.float32)
        buf398 = buf397; del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_69.run(buf398, buf396, primals_199, 58, 25, grid=grid(58), stream=stream0)
        buf399 = reinterpret_tensor(buf390, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70.run(relu_6, buf392, primals_199, primals_29, buf399, 3136, 58, grid=grid(3136, 58), stream=stream0)
        del primals_199
        del primals_29
        del relu_6
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf400 = aten.convolution_backward(buf399, getitem_5, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_5
        del primals_28
        buf401 = buf400[0]
        buf402 = buf400[1]
        del buf400
        buf403 = reinterpret_tensor(buf399, (4, 58, 28, 28), (45472, 784, 28, 1), 0); del buf399  # reuse
        buf408 = reinterpret_tensor(buf392, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_73.run(le_31, buf336, buf352, buf377, buf401, primals_196, primals_26, buf403, buf408, 232, 784, grid=grid(232, 784), stream=stream0)
        del le_31
        del primals_26
        buf404 = empty((58, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_63.run(buf403, buf404, 58, 3136, grid=grid(58), stream=stream0)
        buf405 = reinterpret_tensor(buf396, (58, 25), (25, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_64.run(buf403, convolution_8, primals_195, buf405, 1450, 126, grid=grid(1450), stream=stream0)
        del buf403
        del convolution_8
        del primals_195
        buf406 = empty((58, ), device='cuda', dtype=torch.float32)
        buf407 = buf406; del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_61.run(buf407, buf405, primals_196, 58, 25, grid=grid(58), stream=stream0)
        del primals_196
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf409 = aten.convolution_backward(buf408, add_15, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_15
        del buf408
        del primals_25
        buf410 = buf409[0]
        buf411 = buf409[1]
        del buf409
        buf412 = empty((58, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_63.run(buf410, buf412, 58, 3136, grid=grid(58), stream=stream0)
        buf413 = buf405; del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_64.run(buf410, convolution_7, primals_192, buf413, 1450, 126, grid=grid(1450), stream=stream0)
        del convolution_7
        del primals_192
        buf414 = empty((58, ), device='cuda', dtype=torch.float32)
        buf415 = buf414; del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_61.run(buf415, buf413, primals_193, 58, 25, grid=grid(58), stream=stream0)
        buf416 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_65.run(buf416, primals_193, primals_23, 181888, grid=grid(181888), stream=stream0)
        del primals_193
        del primals_23
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf417 = aten.convolution_backward(buf416, relu_4, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False])
        del primals_22
        buf418 = buf417[0]
        buf419 = buf417[1]
        del buf417
        buf420 = buf413; del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(relu_4, buf418, buf420, 1450, 126, grid=grid(1450), stream=stream0)
        buf421 = empty((58, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf420, buf421, 58, 25, grid=grid(58), stream=stream0)
        buf422 = reinterpret_tensor(buf420, (58, 25), (1, 58), 0); del buf420  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_68.run(relu_4, buf418, convolution_6, primals_189, buf422, 1450, 126, grid=grid(1450), stream=stream0)
        del convolution_6
        del primals_189
        buf423 = empty((58, ), device='cuda', dtype=torch.float32)
        buf424 = buf423; del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_69.run(buf424, buf422, primals_190, 58, 25, grid=grid(58), stream=stream0)
        buf425 = reinterpret_tensor(buf416, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70.run(relu_4, buf418, primals_190, primals_20, buf425, 3136, 58, grid=grid(3136, 58), stream=stream0)
        del buf418
        del primals_190
        del primals_20
        del relu_4
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf426 = aten.convolution_backward(buf425, getitem_3, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf425
        del getitem_3
        del primals_19
        buf427 = buf426[0]
        buf428 = buf426[1]
        del buf426
        buf429 = reinterpret_tensor(buf334, (4, 116, 28, 28), (90944, 784, 28, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf336, buf352, buf377, buf401, buf427, buf429, 363776, grid=grid(363776), stream=stream0)
        del buf336
        del buf352
        del buf377
        del buf401
        buf430 = reinterpret_tensor(buf422, (58, 25), (25, 1), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_75.run(le_33, buf429, buf430, 1450, 126, grid=grid(1450), stream=stream0)
        buf431 = empty((58, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf430, buf431, 58, 25, grid=grid(58), stream=stream0)
        buf432 = reinterpret_tensor(buf430, (58, 25), (1, 58), 0); del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_76.run(le_33, buf429, convolution_5, primals_186, buf432, 1450, 126, grid=grid(1450), stream=stream0)
        del convolution_5
        del primals_186
        buf433 = empty((58, ), device='cuda', dtype=torch.float32)
        buf434 = buf433; del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_69.run(buf434, buf432, primals_187, 58, 25, grid=grid(58), stream=stream0)
        buf435 = reinterpret_tensor(buf427, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_77.run(le_33, buf429, primals_187, primals_17, buf435, 3136, 58, grid=grid(3136, 58), stream=stream0)
        del le_33
        del primals_17
        del primals_187
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf436 = aten.convolution_backward(buf435, add_9, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_9
        del buf435
        del primals_16
        buf437 = buf436[0]
        buf438 = buf436[1]
        del buf436
        buf439 = empty((58, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_63.run(buf437, buf439, 58, 3136, grid=grid(58), stream=stream0)
        buf440 = reinterpret_tensor(buf432, (58, 25), (25, 1), 0); del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_64.run(buf437, convolution_4, primals_183, buf440, 1450, 126, grid=grid(1450), stream=stream0)
        del convolution_4
        del primals_183
        buf441 = empty((58, ), device='cuda', dtype=torch.float32)
        buf442 = buf441; del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_61.run(buf442, buf440, primals_184, 58, 25, grid=grid(58), stream=stream0)
        buf443 = buf437; del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_65.run(buf443, primals_184, primals_14, 181888, grid=grid(181888), stream=stream0)
        del primals_14
        del primals_184
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf444 = aten.convolution_backward(buf443, relu_2, primals_13, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False])
        del primals_13
        buf445 = buf444[0]
        buf446 = buf444[1]
        del buf444
        buf447 = empty((58, 98), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_78.run(relu_2, buf445, buf447, 5684, 128, grid=grid(5684), stream=stream0)
        buf448 = empty((58, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_79.run(buf447, buf448, 58, 98, grid=grid(58), stream=stream0)
        buf449 = reinterpret_tensor(buf447, (58, 98), (1, 58), 0); del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(relu_2, buf445, convolution_3, primals_180, buf449, 5684, 128, grid=grid(5684), stream=stream0)
        del convolution_3
        del primals_180
        buf450 = empty((58, ), device='cuda', dtype=torch.float32)
        buf451 = buf450; del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_81.run(buf451, buf449, primals_181, 58, 98, grid=grid(58), stream=stream0)
        del buf449
        buf452 = empty_strided((4, 58, 56, 56), (181888, 1, 3248, 58), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_82.run(relu_2, buf445, primals_181, primals_11, buf452, 12544, 58, grid=grid(12544, 58), stream=stream0)
        del buf445
        del primals_11
        del primals_181
        del relu_2
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf453 = aten.convolution_backward(buf452, getitem, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf452
        del primals_10
        buf454 = buf453[0]
        buf455 = buf453[1]
        del buf453
        buf456 = buf440; del buf440  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_83.run(le_35, buf429, buf456, 1450, 126, grid=grid(1450), stream=stream0)
        buf457 = empty((58, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf456, buf457, 58, 25, grid=grid(58), stream=stream0)
        buf458 = reinterpret_tensor(buf456, (58, 25), (1, 58), 0); del buf456  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_84.run(le_35, buf429, convolution_2, primals_177, buf458, 1450, 126, grid=grid(1450), stream=stream0)
        del convolution_2
        del primals_177
        buf459 = empty((58, ), device='cuda', dtype=torch.float32)
        buf460 = buf459; del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_69.run(buf460, buf458, primals_178, 58, 25, grid=grid(58), stream=stream0)
        del buf458
        buf461 = reinterpret_tensor(buf443, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_85.run(le_35, buf429, primals_178, primals_8, buf461, 3136, 58, grid=grid(3136, 58), stream=stream0)
        del buf429
        del le_35
        del primals_178
        del primals_8
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf462 = aten.convolution_backward(buf461, add_3, primals_7, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_3
        del buf461
        del primals_7
        buf463 = buf462[0]
        buf464 = buf462[1]
        del buf462
        buf465 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_86.run(buf463, buf465, 24, 3136, grid=grid(24), stream=stream0)
        buf466 = empty((24, 25), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_87.run(buf463, convolution_1, primals_174, buf466, 600, 126, grid=grid(600), stream=stream0)
        del convolution_1
        del primals_174
        buf467 = empty((24, ), device='cuda', dtype=torch.float32)
        buf468 = buf467; del buf467  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_88.run(buf468, buf466, primals_175, 24, 25, grid=grid(24), stream=stream0)
        del buf466
        buf469 = buf463; del buf463  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_89.run(buf469, primals_175, primals_5, 75264, grid=grid(75264), stream=stream0)
        del primals_175
        del primals_5
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf470 = aten.convolution_backward(buf469, getitem, primals_4, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 24, [True, True, False])
        del buf469
        del getitem
        del primals_4
        buf471 = buf470[0]
        buf472 = buf470[1]
        del buf470
        buf473 = buf454; del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(buf473, buf471, 301056, grid=grid(301056), stream=stream0)
        del buf471
        # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
        buf474 = aten.max_pool2d_with_indices_backward(buf473, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_1)
        del buf473
        del getitem_1
        buf475 = buf474
        del buf474
        buf476 = empty_strided((24, 392), (1, 24), device='cuda', dtype=torch.float32)
        buf478 = empty_strided((24, 392), (1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_91.run(relu, buf475, convolution, primals_171, buf476, buf478, 9408, 128, grid=grid(9408), stream=stream0)
        del convolution
        del primals_171
        buf477 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_92.run(buf476, buf477, 24, 392, grid=grid(24), stream=stream0)
        del buf476
        buf479 = empty((24, ), device='cuda', dtype=torch.float32)
        buf480 = buf479; del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_93.run(buf480, buf478, primals_172, 24, 392, grid=grid(24), stream=stream0)
        del buf478
        buf481 = buf475; del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_94.run(buf481, relu, primals_172, primals_2, 1204224, grid=grid(1204224), stream=stream0)
        del primals_172
        del primals_2
        del relu
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf482 = aten.convolution_backward(buf481, primals_339, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf481
        del primals_1
        del primals_339
        buf483 = buf482[1]
        return (buf483, buf480, buf477, buf472, buf468, buf465, buf464, buf460, buf457, buf455, buf451, buf448, buf446, buf442, buf439, buf438, buf434, buf431, buf428, buf424, buf421, buf419, buf415, buf412, buf411, buf407, buf404, buf402, buf398, buf395, buf393, buf389, buf386, buf385, buf381, buf379, buf378, buf374, buf371, buf369, buf365, buf362, buf361, buf357, buf354, buf353, buf349, buf346, buf345, buf341, buf338, buf337, buf333, buf330, buf328, buf324, buf321, buf320, buf316, buf313, buf312, buf308, buf305, buf303, buf299, buf296, buf295, buf291, buf288, buf285, buf281, buf278, buf276, buf272, buf269, buf268, buf264, buf261, buf259, buf255, buf252, buf250, buf246, buf243, buf242, buf238, buf235, buf234, buf230, buf227, buf225, buf221, buf218, buf217, buf213, buf210, buf207, buf203, buf200, buf198, buf194, buf191, buf190, buf186, buf183, buf181, buf177, buf174, buf172, buf168, buf165, buf164, buf160, buf158, buf157, buf153, buf150, buf148, buf144, buf141, buf140, buf136, buf133, buf132, buf128, buf125, buf124, buf120, buf117, buf115, buf111, buf108, buf106, buf102, buf99, buf98, buf94, buf91, buf88, buf84, buf81, buf79, buf75, buf72, buf71, buf67, buf64, buf62, buf58, buf55, buf53, buf49, buf46, buf45, buf41, buf38, buf37, buf33, buf30, buf28, buf24, buf21, buf20, buf16, buf13, buf11, buf7, buf4, reinterpret_tensor(buf1, (1000, 1024), (1024, 1), 0), buf2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((24, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((58, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((58, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((1024, 464, 1, 1), (464, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((4, 24, 112, 112), (301056, 1, 2688, 24), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((4, 24, 112, 112), (301056, 1, 2688, 24), device='cuda:0', dtype=torch.float32)
    getitem = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.int64)
    convolution_1 = rand_strided((4, 24, 28, 28), (18816, 1, 672, 24), device='cuda:0', dtype=torch.float32)
    add_3 = rand_strided((4, 24, 28, 28), (18816, 1, 672, 24), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((4, 58, 56, 56), (181888, 1, 3248, 58), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((4, 58, 56, 56), (181888, 1, 3248, 58), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    add_9 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((4, 58, 28, 28), (90944, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    add_15 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    getitem_5 = rand_strided((4, 58, 28, 28), (90944, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    add_21 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((4, 58, 28, 28), (90944, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    add_27 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.float32)
    view_7 = rand_strided((4, 116, 28, 28), (90944, 1, 3248, 116), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    add_31 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((4, 116, 28, 28), (90944, 1, 3248, 116), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((4, 116, 28, 28), (90944, 1, 3248, 116), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    add_37 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    getitem_9 = rand_strided((4, 116, 14, 14), (45472, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    add_43 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((4, 116, 14, 14), (45472, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    add_49 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((4, 116, 14, 14), (45472, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    add_55 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    getitem_15 = rand_strided((4, 116, 14, 14), (45472, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    add_61 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((4, 116, 14, 14), (45472, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    relu_21 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    add_67 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    getitem_19 = rand_strided((4, 116, 14, 14), (45472, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    relu_23 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    add_73 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    getitem_21 = rand_strided((4, 116, 14, 14), (45472, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    relu_25 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    add_79 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((4, 232, 14, 14), (45472, 1, 3248, 232), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    add_83 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((4, 232, 14, 14), (45472, 1, 3248, 232), device='cuda:0', dtype=torch.float32)
    relu_28 = rand_strided((4, 232, 14, 14), (45472, 1, 3248, 232), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    add_89 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    getitem_23 = rand_strided((4, 232, 7, 7), (22736, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    add_95 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    getitem_25 = rand_strided((4, 232, 7, 7), (22736, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    relu_32 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    add_101 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    getitem_27 = rand_strided((4, 232, 7, 7), (22736, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    relu_34 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    add_107 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.float32)
    view_31 = rand_strided((4, 464, 7, 7), (22736, 1, 3248, 464), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((4, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_17 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda:0', dtype=torch.bool)
    le_1 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.bool)
    le_3 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.bool)
    le_5 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.bool)
    le_7 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.bool)
    le_9 = rand_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda:0', dtype=torch.bool)
    le_10 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.bool)
    le_12 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.bool)
    le_14 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.bool)
    le_16 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.bool)
    le_18 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.bool)
    le_20 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.bool)
    le_22 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.bool)
    le_24 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.bool)
    le_26 = rand_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda:0', dtype=torch.bool)
    le_27 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.bool)
    le_29 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.bool)
    le_31 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.bool)
    le_33 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.bool)
    le_35 = rand_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda:0', dtype=torch.bool)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, primals_316, primals_318, primals_319, primals_321, primals_322, primals_324, primals_325, primals_327, primals_328, primals_330, primals_331, primals_333, primals_334, primals_336, primals_337, primals_339, convolution, relu, getitem, getitem_1, convolution_1, add_3, convolution_2, convolution_3, relu_2, convolution_4, add_9, convolution_5, getitem_3, convolution_6, relu_4, convolution_7, add_15, convolution_8, getitem_5, convolution_9, relu_6, convolution_10, add_21, convolution_11, getitem_7, convolution_12, relu_8, convolution_13, add_27, convolution_14, view_7, convolution_15, add_31, convolution_16, convolution_17, relu_11, convolution_18, add_37, convolution_19, getitem_9, convolution_20, relu_13, convolution_21, add_43, convolution_22, getitem_11, convolution_23, relu_15, convolution_24, add_49, convolution_25, getitem_13, convolution_26, relu_17, convolution_27, add_55, convolution_28, getitem_15, convolution_29, relu_19, convolution_30, add_61, convolution_31, getitem_17, convolution_32, relu_21, convolution_33, add_67, convolution_34, getitem_19, convolution_35, relu_23, convolution_36, add_73, convolution_37, getitem_21, convolution_38, relu_25, convolution_39, add_79, convolution_40, view_23, convolution_41, add_83, convolution_42, convolution_43, relu_28, convolution_44, add_89, convolution_45, getitem_23, convolution_46, relu_30, convolution_47, add_95, convolution_48, getitem_25, convolution_49, relu_32, convolution_50, add_101, convolution_51, getitem_27, convolution_52, relu_34, convolution_53, add_107, convolution_54, view_31, convolution_55, mean, permute_17, le, le_1, le_3, le_5, le_7, le_9, le_10, le_12, le_14, le_16, le_18, le_20, le_22, le_24, le_26, le_27, le_29, le_31, le_33, le_35, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('shufflenet_v2_x1_0', benchmark_compiled_module)
