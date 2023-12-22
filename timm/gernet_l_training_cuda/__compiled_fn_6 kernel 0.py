
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


# kernel path: /tmp/torchinductor_youkaichao/xa/cxawcuodcx7odieikd3nfetxohnim2ksyxfz6lo2jdmlsfuvjuvp.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2560
    x1 = (xindex // 2560)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2560*r2) + (327680*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (x0 + (2560*(r2 // 64)) + (5120*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (2560*r2) + (327680*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 64.0
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


# kernel path: /tmp/torchinductor_youkaichao/qg/cqgm2wye3jb5igzs4drm5rxenufhpm7a4fq42rpqgr4fxyrlriqc.py
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
    size_hints=[4096, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2560*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rc/crcxqtylsieikporxapiiwzucecx2blqvx5nij5wd5ebatvsk4ni.py
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
    size_hints=[4096, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2560*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/og/coggkvd76mzngbjzxtir6tbyxulpreeqpqmxqqfieebefcobk45r.py
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
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 2560
    x2 = (xindex // 163840)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (2560*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = 64.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.001953125
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr0 + (x3), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q4/cq4xew7uowxchqcoh3wb7ongjn5q25hfvvevn7dvmrqpkipdvi7l.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 640
    x1 = (xindex // 640)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (640*r2) + (81920*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((64*x0) + (40960*(r2 // 64)) + (81920*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (640*r2) + (81920*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/mn/cmnkhxhzpybjmyge3admg2i3haio34jc27czrnigty3ms6sfchfo.py
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
    size_hints=[1024, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (640*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sr/csrjedx3fvrsplryctmpawtv6upajdpua62q7ufp4a632fz547jj.py
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
    size_hints=[1024, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (640*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7b/c7b56w77w2ueeow4ejll2zkwysp6onpuzr7mz2bbb6wxs3r76th5.py
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
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 640
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (640*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (64*x2) + (40960*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (640*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.001953125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (640*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33q5quzzo445c3hjp2cjtjz6zi66jn5lutnibwz7un6qhit7tbn.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1920
    x1 = (xindex // 1920)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1920*r2) + (245760*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((64*x0) + (122880*(r2 // 64)) + (245760*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (1920*r2) + (245760*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/js/cjsvxxdhayuithwnqzgs5676qybdxxku25dkw7h2j7se6kghkzkw.py
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
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1920*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e5/ce5wbsrgbhih7hcw6khgrnjufnsjjmave3tnugfhsxu5a4ajlus6.py
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
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1920*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d5/cd5j7evnqcu5xb6zrzs5pxz2lph66t65ou7qpyxt7rxezenhcbkp.py
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
    size_hints=[512, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1920
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (1920*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (64*x2) + (122880*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (1920*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.001953125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (1920*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/in/cinbbtfdhxn6uyovqje7d7b63m275g3q6whneo7hwaeiymkwfsuk.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 640
    x1 = (xindex // 640)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (640*r2) + (81920*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (640*r2) + (81920*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + ((64*x0) + (40960*(r2 // 64)) + (81920*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr3 + ((64*x0) + (40960*(r2 // 64)) + (81920*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr4 + (x0 + (640*r2) + (81920*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tmp3 <= tmp1
        tmp6 = tl.where(tmp4, tmp1, tmp5)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.where(tmp2, tmp1, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp9 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbvlnpn4qi2hrtdvgg6gnkrzuomyf4lfv57dk3deuulzgj2ew2hv.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 640
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (640*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (640*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + (64*x2) + (40960*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (64*x2) + (40960*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x2 + (640*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp6 = tl.where(tmp4, tmp1, tmp5)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.where(tmp2, tmp1, tmp8)
    tmp12 = tmp10 - tmp11
    tmp14 = 0.001953125
    tmp15 = tmp13 * tmp14
    tmp17 = tmp16 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tmp12 * tmp18
    tmp20 = tmp9 - tmp19
    tmp22 = tmp21 * tmp14
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (640*y3)), tmp26, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xi4qdqtkwabcf67newjxx37lduty4auolzl2o4z5crdv2j4hjo.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5120
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 640
    y1 = (yindex // 640)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (640*x2) + (40960*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (640*x2) + (40960*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + (640*x2) + (40960*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp6 = tmp5 <= tmp1
    tmp8 = tl.where(tmp6, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp4, tmp1, tmp10)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.where(tmp2, tmp1, tmp13)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (64*y3)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vx/cvxvq2d5rx6cjzthx27ijcsel7mvvtrbqielu5667zowdal6xsqp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 640
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 64
    r2 = (rindex // 64)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ma/cmaijlttl2rfvmvgxldvr45kmoaw3l7naxaeti23gnrxutd44pqp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 640
    x1 = (xindex // 640)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*x0) + (40960*(r2 // 64)) + (81920*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (640*r2) + (81920*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/le/cle2kv74umx45fzmevd5isz25e542btip47bkuxfvgxdxlcxcciv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 640
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
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (40960*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (640*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.001953125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (640*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jh/cjhlitmmq6abckcjiduhxijpkxvhp73hup2f4hx36c73xexkmmct.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
    xnumel = 640
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex
    r1 = rindex % 64
    r2 = (rindex // 64)
    tmp0 = tl.load(in_ptr0 + (x0 + (640*r3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, other=0.0)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rg/crgsi3nv5xbppbgzyzlnft7u5xpbl6htlln3nvqaz3ejklayqgtw.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 640
    x1 = (xindex // 640)
    tmp8 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (640*r2) + (81920*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((64*x0) + (40960*(r2 // 64)) + (81920*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((64*x0) + (40960*(r2 // 64)) + (81920*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x0 + (640*r2) + (81920*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/mb/cmbyvy67ckhsqmfnyy3x6yd6tlpmya4oomk3p7tqanptvqrjit4p.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 640
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (640*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (64*x2) + (40960*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (64*x2) + (40960*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (640*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp11 = 0.001953125
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2 + (640*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4b/c4bpwpkzfwur4ylixpe65wkva2xduplbnrjku3kj4oteztnt5cc2.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5120
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 640
    y1 = (yindex // 640)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (640*x2) + (40960*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (640*x2) + (40960*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (64*y3)), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bm/cbm4v7pndy5dqbpb6zv3p4d3l7qaq5guve5knn4vltusp4se7tnn.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5120
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 640
    y1 = (yindex // 640)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (640*x2) + (40960*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (640*x2) + (40960*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (64*y3)), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/is/cisyjthu7pq24wedowzjfgekx7nb2lfd2zr4v777zukxuwjk3ltw.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 640
    x1 = (xindex // 640)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*x0) + (40960*(r2 // 64)) + (81920*x1) + (r2 % 64)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (640*r2) + (81920*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (640*r2) + (81920*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ds/cdswlbczmsoi7ta5hr2igtxyylygflg5trphhfnhyzhxofk72uop.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 640
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
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (40960*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (640*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2 + (640*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.001953125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x2 + (640*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (640*y3)), tmp31, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mc/cmc7i3hwd5l2f7absc34ipgsfl7xvnlnw3tjscdnobercccuzwda.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 30720
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (1920*r2) + (245760*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (491520*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/camsrasvpzmrz4ndbz7u5ksguwd5sazwm3bb6ezsk4xcwus3iihg.py
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
    size_hints=[2048, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ei/cei3rkbf75hwhsg4t7q77ch2t4g62ym6vdkrg4dafgmj53xuyu4o.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 30720
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1920
    x1 = (xindex // 1920)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1920*r2) + (245760*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((16*(((r2 + (128*x1)) // 16) % 16)) + (256*x0) + (491520*((r2 + (128*x1)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (1920*r2) + (245760*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gc/cgckqasvkonw7xpxm7gunjkj57df35gzcxab46eqilqunfdob2xj.py
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
    size_hints=[2048, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1920*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bx/cbxtjtdgzoq5s2vxh2bl7deqa567zfjdzidu6v7kpwhu53tkqxj6.py
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
    size_hints=[2048, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1920
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (1920*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (256*x2) + (491520*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (1920*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00048828125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (1920*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ic/cichpoy3uk64eqmf7yyu5y2ej7w6vmuec2la5744wi7bhqhm6vbo.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 2048
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
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (x0 + (640*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (256*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (256*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/cg/ccgatwawt2vvc3ta3jlav7gs66qfzrlm2anah6caj3ehmc22754t.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (640*r2) + (81920*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (163840*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (163840*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (640*r2) + (81920*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5i/c5i6x3cqlzgovay67bn33jqtbsflvvyknyciqxptcebxwuqovtsg.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e4/ce4bewq3fa4mfvucjyejv6om4l45l45kvcb5pwhi2ncvvh5nwr3r.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 640
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (640*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (256*x2) + (163840*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (256*x2) + (163840*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (640*y3)), xmask, eviction_policy='evict_last')
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
    tmp11 = 0.00048828125
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2 + (640*y3)), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qq/cqq6zqgrtaapmjfoiru3dcf66azorxja4f35kg5mfdrg33klu67g.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (160*r2) + (20480*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (40960*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cc/cccqydtr77r4qdzfedti7gn6732atpq377jrl3gjvwnuoijhbett.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qg/cqgzt3hvxiy35x5sgrtdi3wjs2tddgccoixfvk4ecxrbpuj2yp77.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2560
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
        tmp3 = tl.load(in_ptr1 + ((16*(((r2 + (128*x1)) // 16) % 16)) + (256*x0) + (40960*((r2 + (128*x1)) // 256)) + (r2 % 16)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/al/calcuqobp7t6ec62zkmvlhqkemqznvxrxsn4l5v7aaom3zle47gv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 16
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/6f/c6figbrsce66mtxc5m52f5i3txuzn3v65df4bmfqjrd24kidrsfy.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 160
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (160*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (256*x2) + (40960*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (160*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00048828125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (160*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jp/cjpm7giv7gazgu3e7yap36ydt4h76jisoxhnjuphn2wylvgowlv6.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_40', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5120
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 640
    y1 = (yindex // 640)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (640*x2) + (163840*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (640*x2) + (163840*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (256*y3)), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2h/c2hoiretmqrlyaw2g6bsd44743abx6qyashdkq5kxktiaxejvviz.py
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
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (163840*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/56/c563uwfhltgevgaoorih3ocqnwfk6z5vrdlajswkxealxdecqg3u.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (163840*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (640*r2) + (81920*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7q/c7qjdxegicnwjwvskshqtxc7x6644ukdjpylgnjspkqopqrihhcv.py
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
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 640
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
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (163840*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (640*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00048828125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (640*y3)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ql/cqlbs4m7pqav5c4afazc7kbhbyfqjxvuor73s3z2ykgygtcnl3aw.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (163840*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (640*r2) + (81920*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (640*r2) + (81920*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i6/ci6zyczf4zwsgqqy6tsgn5rg4y2wrqsvcp6tp5qsp6nvfubfmh42.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 640
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
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (163840*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (640*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2 + (640*y3)), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00048828125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x2 + (640*y3)), tmp17, xmask)
    tl.store(out_ptr1 + (x2 + (640*y3)), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/he/chesmowbwdofj2sdoslonguyry7svlkl6ep3w5whu4q65ta2tkxo.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (160*r2) + (20480*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (163840*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5vsaxky4y5ez5se6mmmk62efaxefwxttvszninhjtgvqvh2kwy.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lw/clwnywjrt3xpkefoseagthnvfzgcmqtvropr75g36tkn55rql55p.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 160
    x1 = (xindex // 160)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (160*r2) + (20480*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((32*(((r2 + (128*x1)) // 32) % 32)) + (1024*x0) + (163840*((r2 + (128*x1)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (160*r2) + (20480*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ft/cftait7fmqahi66zoi6cd42fe7suloqvb6kpbgfytui4gh5xov35.py
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
    size_hints=[256, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 64
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


# kernel path: /tmp/torchinductor_youkaichao/xq/cxqxuyc4hrcdl33ebgbdwey2qfvhknzc33kmexpqm65i73xgst7e.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 160
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x2 + (160*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (1024*x2) + (163840*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (160*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0001220703125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (160*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tm/ctmwwc3htpframemjrqmazt7zsga5m2qt7mbpkawg2hiictsqeaf.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 8192
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
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (1024*x0) + (196608*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (1024*x0) + (196608*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qj/cqj2o4f4qa2zz6zrignrfyexe5rupbht42yz53lcgatrkdnnsh7s.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (192*r2) + (24576*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (196608*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (196608*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (192*r2) + (24576*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r2/cr223ntjirri5axx6hlocmhfljq6jdntkbfow35edc37bynfm4e3.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/es/ceschzk4iyyq54o7sc3khu7mbyxygrhkeyzapyot2lmzibziwdbw.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (1024*x2) + (196608*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (1024*x2) + (196608*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (192*y3)), xmask, eviction_policy='evict_last')
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
    tmp11 = 0.0001220703125
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ok/cokdo3heoapfmprlo3ymatnzbgecap5mn5qex2lfxzvhemowokup.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (192*r2) + (24576*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (196608*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lv/clv7zep4cqd7smh5ixpz6onghrv22q2cxlt4psgksdlgxols6zyw.py
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
    size_hints=[256, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m6/cm6ryi66onf65bokeldcup3yju7pqhsxmo2dqn7fh2442qkhovi5.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (24576*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((32*(((r2 + (128*x1)) // 32) % 32)) + (1024*x0) + (196608*((r2 + (128*x1)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (192*r2) + (24576*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2g/c2gzlsc5lggx6qmq6tvt7sze44kjpac5eeje5joomclxu5ngpe7r.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 64
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/cf/ccfa5dykxsnsmh6lxnlsiskwu6aryamg2a6tqqtpp5fvi4odjdmv.py
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
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (1024*x2) + (196608*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0001220703125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v6/cv6dsn5bkdpgbmczskjoz77h3ca5debluue47odygdcznmk54yob.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_60', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (196608*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (192*x2) + (196608*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (1024*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vb/cvbq2pgpt3yjq3rqsqgzuxucoubnsbx4moc33qoatqutpaef7x2n.py
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
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (196608*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oq/coqdgvbgpe34wiuc57akrmqsu27g6y5sxgd3byo247y56yp7ftqg.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (196608*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (192*r2) + (24576*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (192*r2) + (24576*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gd/cgdyyuwrrvbec2l7te4u7wc4wzg6mjoq654q55abod4jolzh3nzn.py
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
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x2) + (196608*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (192*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2 + (192*y3)), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0001220703125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp17, xmask)
    tl.store(out_ptr1 + (x2 + (192*y3)), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/np/cnp777ewkvro7zrfgzckjrhlcq4tllvpfyumygqhlz5e46awarso.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (1048576*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ae/cael3ugmxugva7bq5s3e6fukmsprax44ajkq5nksrnt7j76dwsqc.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_65', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/bj/cbjmdaiiqbq55tsxw3dnwklnrtncd6675tmqhyvygwuhvq2knoxr.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (524288*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (524288*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (128*r2) + (16384*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr5 + (x1 + (128*r2) + (16384*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp6 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ft/cft4bksucdxehki7pzxgb5vsqcinq6vtoe3w62fd4lbt2doswuqx.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hv/chvhcn7uxt7bfdi6gevxe2wfajyrlbgk6kct6mbubrpnbjogo4f2.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16, 17))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 4096
    y1 = (yindex // 4096)
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (4096*x2) + (524288*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (4096*x2) + (524288*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr12 + (x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 3.0517578125e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp26 = tmp24 - tmp25
    tmp28 = tmp27 * tmp11
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 * tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = tmp6 - tmp32
    tmp34 = tmp33 - tmp19
    tmp36 = tmp29 * tmp35
    tmp37 = tmp34 * tmp36
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp23, xmask)
    tl.store(out_ptr1 + (x2 + (128*y3)), tmp37, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yw/cywarawk4v5eef6wqe4xhnxnsbgeq2lvajlir3rfpqwsha47eoh5.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (524288*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pm/cpmun2rfrg2qlanmk4rsxknz2netwmqotri7w35bl2bzgdr4nlx7.py
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
    size_hints=[128, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m3/cm3r7eq47dbdlwlavafycpx6f3kg3dhctvgwbcl23lnvt6xznjkp.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((64*(((r2 + (128*x1)) // 64) % 64)) + (4096*x0) + (524288*((r2 + (128*x1)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q2/cq2bkmnqvna3shrj5wnxcoca3exfizbvb7v2uj3qfeaso3x5c3vx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 256
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


# kernel path: /tmp/torchinductor_youkaichao/p6/cp6jdwtmljvvyrbrypugs74vdxeb5p5vff73szszr7bq4jnw34pd.py
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
    size_hints=[32768, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 4096
    y1 = (yindex // 4096)
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (4096*x2) + (524288*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 3.0517578125e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7j/c7j7gptzvudyhqapko5mu27xdjfyc3jn32orxjvqtvjrawbqm2yh.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (32*r2) + (262144*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (524288*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (524288*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/gj/cgjegwps62lkssxmgbyg27zrblt7rnx7eqlsuyhv5pr4vhb5mvv3.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wc/cwcgvl4kp74qaalahp5equabalpm7q4rnh2vytxgtpxf6ulnfyyb.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (32*r2) + (4096*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r2 + (128*(x0 % 128)) + (16384*x1) + (524288*((r2 + (128*x0)) // 16384))), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r2 + (128*(x0 % 128)) + (16384*x1) + (524288*((r2 + (128*x0)) // 16384))), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (32*r2) + (4096*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q6/cq6kcyah3r6onbxkubnpu2wsgkfu35ffhqxzigvgonh2ssdzq2ac.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 32
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zw/czw3lgljhkfn5h4etvlkjlskct7yfrusruglaqq6x3xfrkdwhgrv.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_78 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16384
    y1 = (yindex // 16384)
    tmp0 = tl.load(in_ptr0 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (16384*x2) + (524288*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (16384*x2) + (524288*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
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
    tmp11 = 7.62939453125e-06
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2 + (32*y3)), tmp23, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_345, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, convolution_3, squeeze_10, relu_2, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_4, convolution_7, squeeze_22, relu_5, convolution_8, squeeze_25, relu_6, convolution_9, squeeze_28, relu_7, convolution_10, squeeze_31, relu_8, convolution_11, squeeze_34, convolution_12, squeeze_37, relu_9, convolution_13, squeeze_40, relu_10, convolution_14, squeeze_43, relu_11, convolution_15, squeeze_46, relu_12, convolution_16, squeeze_49, relu_13, convolution_17, squeeze_52, relu_14, convolution_18, squeeze_55, relu_15, convolution_19, squeeze_58, relu_16, convolution_20, squeeze_61, relu_17, convolution_21, squeeze_64, relu_18, convolution_22, squeeze_67, relu_19, convolution_23, squeeze_70, relu_20, convolution_24, squeeze_73, relu_21, convolution_25, squeeze_76, relu_22, convolution_26, squeeze_79, relu_23, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, convolution_31, squeeze_94, relu_27, convolution_32, squeeze_97, relu_28, convolution_33, squeeze_100, relu_29, convolution_34, squeeze_103, relu_30, convolution_35, squeeze_106, relu_31, convolution_36, squeeze_109, relu_32, convolution_37, squeeze_112, relu_33, convolution_38, squeeze_115, relu_34, convolution_39, squeeze_118, relu_35, convolution_40, squeeze_121, relu_36, convolution_41, squeeze_124, relu_37, convolution_42, squeeze_127, relu_38, convolution_43, squeeze_130, relu_39, convolution_44, squeeze_133, relu_40, convolution_45, squeeze_136, relu_41, convolution_46, squeeze_139, relu_42, convolution_47, squeeze_142, relu_43, convolution_48, squeeze_145, relu_44, convolution_49, squeeze_148, relu_45, convolution_50, squeeze_151, relu_46, convolution_51, squeeze_154, relu_47, convolution_52, squeeze_157, relu_48, convolution_53, squeeze_160, relu_49, convolution_54, squeeze_163, relu_50, convolution_55, squeeze_166, relu_51, convolution_56, squeeze_169, clone, permute_1, le, unsqueeze_230, unsqueeze_242, unsqueeze_254, unsqueeze_266, unsqueeze_278, unsqueeze_290, unsqueeze_302, unsqueeze_314, unsqueeze_326, unsqueeze_338, unsqueeze_350, unsqueeze_362, unsqueeze_374, unsqueeze_386, unsqueeze_398, unsqueeze_410, unsqueeze_422, unsqueeze_434, unsqueeze_446, unsqueeze_458, unsqueeze_470, unsqueeze_482, unsqueeze_494, unsqueeze_506, unsqueeze_518, unsqueeze_530, unsqueeze_542, unsqueeze_554, unsqueeze_566, unsqueeze_578, unsqueeze_590, unsqueeze_602, unsqueeze_614, unsqueeze_626, unsqueeze_638, unsqueeze_650, unsqueeze_662, unsqueeze_674, unsqueeze_686, unsqueeze_698, unsqueeze_710, unsqueeze_722, unsqueeze_734, unsqueeze_746, unsqueeze_758, unsqueeze_770, unsqueeze_782, unsqueeze_794, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_9, (192, ), (1, ))
    assert_size_stride(primals_11, (192, ), (1, ))
    assert_size_stride(primals_13, (192, ), (1, ))
    assert_size_stride(primals_15, (192, ), (1, ))
    assert_size_stride(primals_17, (192, ), (1, ))
    assert_size_stride(primals_19, (160, ), (1, ))
    assert_size_stride(primals_21, (160, ), (1, ))
    assert_size_stride(primals_23, (640, ), (1, ))
    assert_size_stride(primals_25, (640, ), (1, ))
    assert_size_stride(primals_27, (160, ), (1, ))
    assert_size_stride(primals_29, (160, ), (1, ))
    assert_size_stride(primals_31, (640, ), (1, ))
    assert_size_stride(primals_33, (160, ), (1, ))
    assert_size_stride(primals_35, (160, ), (1, ))
    assert_size_stride(primals_37, (640, ), (1, ))
    assert_size_stride(primals_39, (160, ), (1, ))
    assert_size_stride(primals_41, (160, ), (1, ))
    assert_size_stride(primals_43, (640, ), (1, ))
    assert_size_stride(primals_45, (160, ), (1, ))
    assert_size_stride(primals_47, (160, ), (1, ))
    assert_size_stride(primals_49, (640, ), (1, ))
    assert_size_stride(primals_51, (160, ), (1, ))
    assert_size_stride(primals_53, (160, ), (1, ))
    assert_size_stride(primals_55, (640, ), (1, ))
    assert_size_stride(primals_57, (1920, ), (1, ))
    assert_size_stride(primals_59, (1920, ), (1, ))
    assert_size_stride(primals_61, (640, ), (1, ))
    assert_size_stride(primals_63, (640, ), (1, ))
    assert_size_stride(primals_65, (1920, ), (1, ))
    assert_size_stride(primals_67, (1920, ), (1, ))
    assert_size_stride(primals_69, (640, ), (1, ))
    assert_size_stride(primals_71, (1920, ), (1, ))
    assert_size_stride(primals_73, (1920, ), (1, ))
    assert_size_stride(primals_75, (640, ), (1, ))
    assert_size_stride(primals_77, (1920, ), (1, ))
    assert_size_stride(primals_79, (1920, ), (1, ))
    assert_size_stride(primals_81, (640, ), (1, ))
    assert_size_stride(primals_83, (1920, ), (1, ))
    assert_size_stride(primals_85, (1920, ), (1, ))
    assert_size_stride(primals_87, (640, ), (1, ))
    assert_size_stride(primals_89, (1920, ), (1, ))
    assert_size_stride(primals_91, (1920, ), (1, ))
    assert_size_stride(primals_93, (640, ), (1, ))
    assert_size_stride(primals_95, (1920, ), (1, ))
    assert_size_stride(primals_97, (1920, ), (1, ))
    assert_size_stride(primals_99, (640, ), (1, ))
    assert_size_stride(primals_101, (1920, ), (1, ))
    assert_size_stride(primals_103, (1920, ), (1, ))
    assert_size_stride(primals_105, (640, ), (1, ))
    assert_size_stride(primals_107, (1920, ), (1, ))
    assert_size_stride(primals_109, (1920, ), (1, ))
    assert_size_stride(primals_111, (640, ), (1, ))
    assert_size_stride(primals_113, (2560, ), (1, ))
    assert_size_stride(primals_115, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_116, (128, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_117, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_118, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_119, (192, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_120, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_121, (192, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_122, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_123, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_124, (160, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_125, (160, 160, 3, 3), (1440, 1, 480, 160))
    assert_size_stride(primals_126, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_127, (640, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_128, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_129, (160, 160, 3, 3), (1440, 1, 480, 160))
    assert_size_stride(primals_130, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_131, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_132, (160, 160, 3, 3), (1440, 1, 480, 160))
    assert_size_stride(primals_133, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_134, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_135, (160, 160, 3, 3), (1440, 1, 480, 160))
    assert_size_stride(primals_136, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_137, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_138, (160, 160, 3, 3), (1440, 1, 480, 160))
    assert_size_stride(primals_139, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_140, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_141, (160, 160, 3, 3), (1440, 1, 480, 160))
    assert_size_stride(primals_142, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_143, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_144, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_145, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_146, (640, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_147, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_148, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_149, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_150, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_151, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_152, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_153, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_154, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_155, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_156, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_157, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_158, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_159, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_160, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_161, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_162, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_163, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_164, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_165, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_166, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_167, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_168, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_169, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_170, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_171, (2560, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_345, (8, 3, 256, 256), (196608, 1, 768, 3))
    assert_size_stride(convolution, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(relu, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(convolution_1, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(squeeze_4, (128, ), (1, ))
    assert_size_stride(relu_1, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(convolution_2, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(squeeze_7, (128, ), (1, ))
    assert_size_stride(convolution_3, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(squeeze_10, (128, ), (1, ))
    assert_size_stride(relu_2, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(convolution_4, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(squeeze_13, (192, ), (1, ))
    assert_size_stride(relu_3, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(convolution_5, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(squeeze_16, (192, ), (1, ))
    assert_size_stride(convolution_6, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(squeeze_19, (192, ), (1, ))
    assert_size_stride(relu_4, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(convolution_7, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(squeeze_22, (192, ), (1, ))
    assert_size_stride(relu_5, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(convolution_8, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(squeeze_25, (192, ), (1, ))
    assert_size_stride(relu_6, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(convolution_9, (8, 160, 32, 32), (163840, 1, 5120, 160))
    assert_size_stride(squeeze_28, (160, ), (1, ))
    assert_size_stride(relu_7, (8, 160, 32, 32), (163840, 1, 5120, 160))
    assert_size_stride(convolution_10, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_31, (160, ), (1, ))
    assert_size_stride(relu_8, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_11, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(squeeze_34, (640, ), (1, ))
    assert_size_stride(convolution_12, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(squeeze_37, (640, ), (1, ))
    assert_size_stride(relu_9, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(convolution_13, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_40, (160, ), (1, ))
    assert_size_stride(relu_10, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_14, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_43, (160, ), (1, ))
    assert_size_stride(relu_11, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_15, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(squeeze_46, (640, ), (1, ))
    assert_size_stride(relu_12, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(convolution_16, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_49, (160, ), (1, ))
    assert_size_stride(relu_13, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_17, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_52, (160, ), (1, ))
    assert_size_stride(relu_14, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_18, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(squeeze_55, (640, ), (1, ))
    assert_size_stride(relu_15, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(convolution_19, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_58, (160, ), (1, ))
    assert_size_stride(relu_16, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_20, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_61, (160, ), (1, ))
    assert_size_stride(relu_17, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_21, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(squeeze_64, (640, ), (1, ))
    assert_size_stride(relu_18, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(convolution_22, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_67, (160, ), (1, ))
    assert_size_stride(relu_19, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_23, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_70, (160, ), (1, ))
    assert_size_stride(relu_20, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_24, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(squeeze_73, (640, ), (1, ))
    assert_size_stride(relu_21, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(convolution_25, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_76, (160, ), (1, ))
    assert_size_stride(relu_22, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_26, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_79, (160, ), (1, ))
    assert_size_stride(relu_23, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_27, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(squeeze_82, (640, ), (1, ))
    assert_size_stride(relu_24, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(convolution_28, (8, 1920, 16, 16), (491520, 1, 30720, 1920))
    assert_size_stride(squeeze_85, (1920, ), (1, ))
    assert_size_stride(relu_25, (8, 1920, 16, 16), (491520, 1, 30720, 1920))
    assert_size_stride(convolution_29, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_88, (1920, ), (1, ))
    assert_size_stride(relu_26, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_30, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_91, (640, ), (1, ))
    assert_size_stride(convolution_31, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_94, (640, ), (1, ))
    assert_size_stride(relu_27, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(convolution_32, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_97, (1920, ), (1, ))
    assert_size_stride(relu_28, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_33, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_100, (1920, ), (1, ))
    assert_size_stride(relu_29, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_34, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_103, (640, ), (1, ))
    assert_size_stride(relu_30, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(convolution_35, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_106, (1920, ), (1, ))
    assert_size_stride(relu_31, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_36, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_109, (1920, ), (1, ))
    assert_size_stride(relu_32, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_37, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_112, (640, ), (1, ))
    assert_size_stride(relu_33, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(convolution_38, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_115, (1920, ), (1, ))
    assert_size_stride(relu_34, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_39, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_118, (1920, ), (1, ))
    assert_size_stride(relu_35, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_40, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_121, (640, ), (1, ))
    assert_size_stride(relu_36, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(convolution_41, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_124, (1920, ), (1, ))
    assert_size_stride(relu_37, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_42, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_127, (1920, ), (1, ))
    assert_size_stride(relu_38, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_43, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_130, (640, ), (1, ))
    assert_size_stride(relu_39, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(convolution_44, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_133, (1920, ), (1, ))
    assert_size_stride(relu_40, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_45, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_136, (1920, ), (1, ))
    assert_size_stride(relu_41, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_46, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_139, (640, ), (1, ))
    assert_size_stride(relu_42, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(convolution_47, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_142, (1920, ), (1, ))
    assert_size_stride(relu_43, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_48, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_145, (1920, ), (1, ))
    assert_size_stride(relu_44, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_49, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_148, (640, ), (1, ))
    assert_size_stride(relu_45, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(convolution_50, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_151, (1920, ), (1, ))
    assert_size_stride(relu_46, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_51, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_154, (1920, ), (1, ))
    assert_size_stride(relu_47, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_52, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_157, (640, ), (1, ))
    assert_size_stride(relu_48, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(convolution_53, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_160, (1920, ), (1, ))
    assert_size_stride(relu_49, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_54, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_163, (1920, ), (1, ))
    assert_size_stride(relu_50, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_55, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_166, (640, ), (1, ))
    assert_size_stride(relu_51, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(convolution_56, (8, 2560, 8, 8), (163840, 1, 20480, 2560))
    assert_size_stride(squeeze_169, (2560, ), (1, ))
    assert_size_stride(clone, (8, 2560), (2560, 1))
    assert_size_stride(permute_1, (1000, 2560), (2560, 1))
    assert_size_stride(le, (8, 2560, 8, 8), (163840, 1, 20480, 2560))
    assert_size_stride(unsqueeze_230, (1, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(unsqueeze_242, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_254, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_266, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_278, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_290, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_302, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_314, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_326, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_338, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_350, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_362, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_374, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_386, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_398, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_410, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_422, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_434, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_446, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_458, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_470, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_482, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_494, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_506, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_518, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_530, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_542, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_554, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_566, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_578, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_590, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_602, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_614, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_626, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_638, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_650, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_662, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_674, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_686, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_698, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_710, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_722, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_734, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_746, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_758, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_770, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_782, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_794, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_806, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_818, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_830, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_842, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_854, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_866, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_878, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_890, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_902, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 2560), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 2560), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone, out=buf1)
        del clone
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty_strided((2560, 4), (1, 2560), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((2560, 4), (1, 2560), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_div_native_batch_norm_backward_threshold_backward_1.run(le, buf0, convolution_56, unsqueeze_230, buf3, buf5, 10240, 128, grid=grid(10240), stream=stream0)
        buf4 = empty((2560, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_2.run(buf3, buf4, 2560, 4, grid=grid(2560), stream=stream0)
        buf6 = empty((2560, ), device='cuda', dtype=torch.float32)
        buf7 = empty((2560, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_3.run(buf5, squeeze_169, buf6, buf7, 2560, 4, grid=grid(2560), stream=stream0)
        buf8 = empty_strided((8, 2560, 8, 8), (163840, 1, 20480, 2560), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4.run(le, buf0, convolution_56, unsqueeze_230, buf6, squeeze_169, buf4, primals_113, buf8, 1310720, grid=grid(1310720), stream=stream0)
        del buf0
        del convolution_56
        del le
        del primals_113
        del squeeze_169
        del unsqueeze_230
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf9 = aten.convolution_backward(buf8, relu_51, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_171
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        buf12 = reinterpret_tensor(buf6, (640, 4), (1, 640), 0); del buf6  # reuse
        buf14 = empty_strided((640, 4), (1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(relu_51, buf10, convolution_55, unsqueeze_242, buf12, buf14, 2560, 128, grid=grid(2560), stream=stream0)
        buf13 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(buf12, buf13, 640, 4, grid=grid(640), stream=stream0)
        buf15 = empty((640, ), device='cuda', dtype=torch.float32)
        buf16 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf14, squeeze_166, buf15, buf16, 640, 4, grid=grid(640), stream=stream0)
        buf17 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8.run(relu_51, buf10, convolution_55, unsqueeze_242, buf15, squeeze_166, buf13, primals_111, buf17, 512, 640, grid=grid(512, 640), stream=stream0)
        del convolution_55
        del primals_111
        del squeeze_166
        del unsqueeze_242
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf18 = aten.convolution_backward(buf17, relu_50, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_170
        buf19 = buf18[0]
        buf20 = buf18[1]
        del buf18
        buf21 = empty_strided((1920, 4), (1, 1920), device='cuda', dtype=torch.float32)
        buf23 = empty_strided((1920, 4), (1, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_50, buf19, convolution_54, unsqueeze_254, buf21, buf23, 7680, 128, grid=grid(7680), stream=stream0)
        buf22 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf21, buf22, 1920, 4, grid=grid(1920), stream=stream0)
        buf24 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf25 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf23, squeeze_163, buf24, buf25, 1920, 4, grid=grid(1920), stream=stream0)
        buf26 = empty_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_50, buf19, convolution_54, unsqueeze_254, buf24, squeeze_163, buf22, primals_109, buf26, 512, 1920, grid=grid(512, 1920), stream=stream0)
        del buf19
        del convolution_54
        del primals_109
        del relu_50
        del squeeze_163
        del unsqueeze_254
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf27 = aten.convolution_backward(buf26, relu_49, primals_169, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
        del primals_169
        buf28 = buf27[0]
        buf29 = buf27[1]
        del buf27
        buf30 = buf23; del buf23  # reuse
        buf32 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_49, buf28, convolution_53, unsqueeze_266, buf30, buf32, 7680, 128, grid=grid(7680), stream=stream0)
        buf31 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf30, buf31, 1920, 4, grid=grid(1920), stream=stream0)
        buf33 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf34 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf32, squeeze_160, buf33, buf34, 1920, 4, grid=grid(1920), stream=stream0)
        buf35 = buf26; del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_49, buf28, convolution_53, unsqueeze_266, buf33, squeeze_160, buf31, primals_107, buf35, 512, 1920, grid=grid(512, 1920), stream=stream0)
        del buf28
        del convolution_53
        del primals_107
        del relu_49
        del squeeze_160
        del unsqueeze_266
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf36 = aten.convolution_backward(buf35, relu_48, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_168
        buf37 = buf36[0]
        buf38 = buf36[1]
        del buf36
        buf39 = buf14; del buf14  # reuse
        buf41 = buf12; del buf12  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_13.run(relu_48, relu_51, buf10, buf37, convolution_52, unsqueeze_278, buf39, buf41, 2560, 128, grid=grid(2560), stream=stream0)
        buf40 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(buf39, buf40, 640, 4, grid=grid(640), stream=stream0)
        buf42 = empty((640, ), device='cuda', dtype=torch.float32)
        buf44 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf41, squeeze_157, buf42, buf44, 640, 4, grid=grid(640), stream=stream0)
        buf43 = buf17; del buf17  # reuse
        buf45 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_14.run(buf45, relu_48, relu_51, buf10, buf37, convolution_52, unsqueeze_278, buf42, squeeze_157, buf40, primals_105, 512, 640, grid=grid(512, 640), stream=stream0)
        del convolution_52
        del primals_105
        del squeeze_157
        del unsqueeze_278
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf46 = aten.convolution_backward(buf45, relu_47, primals_167, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf45
        del primals_167
        buf47 = buf46[0]
        buf48 = buf46[1]
        del buf46
        buf49 = buf32; del buf32  # reuse
        buf51 = buf30; del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_47, buf47, convolution_51, unsqueeze_290, buf49, buf51, 7680, 128, grid=grid(7680), stream=stream0)
        buf50 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf49, buf50, 1920, 4, grid=grid(1920), stream=stream0)
        buf52 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf53 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf51, squeeze_154, buf52, buf53, 1920, 4, grid=grid(1920), stream=stream0)
        buf54 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_47, buf47, convolution_51, unsqueeze_290, buf52, squeeze_154, buf50, primals_103, buf54, 512, 1920, grid=grid(512, 1920), stream=stream0)
        del buf47
        del convolution_51
        del primals_103
        del relu_47
        del squeeze_154
        del unsqueeze_290
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf55 = aten.convolution_backward(buf54, relu_46, primals_166, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
        del primals_166
        buf56 = buf55[0]
        buf57 = buf55[1]
        del buf55
        buf58 = buf51; del buf51  # reuse
        buf60 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_46, buf56, convolution_50, unsqueeze_302, buf58, buf60, 7680, 128, grid=grid(7680), stream=stream0)
        buf59 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf58, buf59, 1920, 4, grid=grid(1920), stream=stream0)
        buf61 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf62 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf60, squeeze_151, buf61, buf62, 1920, 4, grid=grid(1920), stream=stream0)
        buf63 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_46, buf56, convolution_50, unsqueeze_302, buf61, squeeze_151, buf59, primals_101, buf63, 512, 1920, grid=grid(512, 1920), stream=stream0)
        del buf56
        del convolution_50
        del primals_101
        del relu_46
        del squeeze_151
        del unsqueeze_302
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf64 = aten.convolution_backward(buf63, relu_45, primals_165, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_165
        buf65 = buf64[0]
        buf66 = buf64[1]
        del buf64
        buf67 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_15.run(buf67, relu_45, relu_48, relu_51, buf37, buf65, 5120, 64, grid=grid(5120, 64), stream=stream0)
        del buf37
        del relu_45
        del relu_48
        del relu_51
        buf68 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_16.run(buf67, buf68, 640, 512, grid=grid(640), stream=stream0)
        buf69 = buf41; del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_17.run(buf67, convolution_49, unsqueeze_314, buf69, 2560, 128, grid=grid(2560), stream=stream0)
        buf70 = empty((640, ), device='cuda', dtype=torch.float32)
        buf71 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf69, squeeze_148, buf70, buf71, 640, 4, grid=grid(640), stream=stream0)
        buf72 = reinterpret_tensor(buf65, (8, 640, 8, 8), (40960, 1, 5120, 640), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_18.run(buf67, convolution_49, unsqueeze_314, buf70, squeeze_148, buf68, primals_99, buf72, 512, 640, grid=grid(512, 640), stream=stream0)
        del convolution_49
        del primals_99
        del squeeze_148
        del unsqueeze_314
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf73 = aten.convolution_backward(buf72, relu_44, primals_164, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_164
        buf74 = buf73[0]
        buf75 = buf73[1]
        del buf73
        buf76 = buf60; del buf60  # reuse
        buf78 = buf58; del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_44, buf74, convolution_48, unsqueeze_326, buf76, buf78, 7680, 128, grid=grid(7680), stream=stream0)
        buf77 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf76, buf77, 1920, 4, grid=grid(1920), stream=stream0)
        buf79 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf80 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf78, squeeze_145, buf79, buf80, 1920, 4, grid=grid(1920), stream=stream0)
        buf81 = buf63; del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_44, buf74, convolution_48, unsqueeze_326, buf79, squeeze_145, buf77, primals_97, buf81, 512, 1920, grid=grid(512, 1920), stream=stream0)
        del buf74
        del convolution_48
        del primals_97
        del relu_44
        del squeeze_145
        del unsqueeze_326
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf82 = aten.convolution_backward(buf81, relu_43, primals_163, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
        del primals_163
        buf83 = buf82[0]
        buf84 = buf82[1]
        del buf82
        buf85 = buf78; del buf78  # reuse
        buf87 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_43, buf83, convolution_47, unsqueeze_338, buf85, buf87, 7680, 128, grid=grid(7680), stream=stream0)
        buf86 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf85, buf86, 1920, 4, grid=grid(1920), stream=stream0)
        buf88 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf89 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf87, squeeze_142, buf88, buf89, 1920, 4, grid=grid(1920), stream=stream0)
        buf90 = buf81; del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_43, buf83, convolution_47, unsqueeze_338, buf88, squeeze_142, buf86, primals_95, buf90, 512, 1920, grid=grid(512, 1920), stream=stream0)
        del buf83
        del convolution_47
        del primals_95
        del relu_43
        del squeeze_142
        del unsqueeze_338
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf91 = aten.convolution_backward(buf90, relu_42, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_162
        buf92 = buf91[0]
        buf93 = buf91[1]
        del buf91
        buf94 = buf70; del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_19.run(relu_42, buf67, buf92, buf94, 640, 512, grid=grid(640), stream=stream0)
        buf95 = buf69; del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_20.run(relu_42, buf67, buf92, convolution_46, unsqueeze_350, buf95, 2560, 128, grid=grid(2560), stream=stream0)
        buf96 = empty((640, ), device='cuda', dtype=torch.float32)
        buf98 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf95, squeeze_139, buf96, buf98, 640, 4, grid=grid(640), stream=stream0)
        buf97 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_21.run(relu_42, buf67, buf92, convolution_46, unsqueeze_350, buf96, squeeze_139, buf94, primals_93, buf97, 512, 640, grid=grid(512, 640), stream=stream0)
        del convolution_46
        del primals_93
        del squeeze_139
        del unsqueeze_350
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf99 = aten.convolution_backward(buf97, relu_41, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf97
        del primals_161
        buf100 = buf99[0]
        buf101 = buf99[1]
        del buf99
        buf102 = buf87; del buf87  # reuse
        buf104 = buf85; del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_41, buf100, convolution_45, unsqueeze_362, buf102, buf104, 7680, 128, grid=grid(7680), stream=stream0)
        buf103 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf102, buf103, 1920, 4, grid=grid(1920), stream=stream0)
        buf105 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf106 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf104, squeeze_136, buf105, buf106, 1920, 4, grid=grid(1920), stream=stream0)
        buf107 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_41, buf100, convolution_45, unsqueeze_362, buf105, squeeze_136, buf103, primals_91, buf107, 512, 1920, grid=grid(512, 1920), stream=stream0)
        del buf100
        del convolution_45
        del primals_91
        del relu_41
        del squeeze_136
        del unsqueeze_362
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf108 = aten.convolution_backward(buf107, relu_40, primals_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
        del primals_160
        buf109 = buf108[0]
        buf110 = buf108[1]
        del buf108
        buf111 = buf104; del buf104  # reuse
        buf113 = buf102; del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_40, buf109, convolution_44, unsqueeze_374, buf111, buf113, 7680, 128, grid=grid(7680), stream=stream0)
        buf112 = buf105; del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf111, buf112, 1920, 4, grid=grid(1920), stream=stream0)
        buf114 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf115 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf113, squeeze_133, buf114, buf115, 1920, 4, grid=grid(1920), stream=stream0)
        buf116 = buf107; del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_40, buf109, convolution_44, unsqueeze_374, buf114, squeeze_133, buf112, primals_89, buf116, 512, 1920, grid=grid(512, 1920), stream=stream0)
        del buf109
        del convolution_44
        del primals_89
        del relu_40
        del squeeze_133
        del unsqueeze_374
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf117 = aten.convolution_backward(buf116, relu_39, primals_159, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_159
        buf118 = buf117[0]
        buf119 = buf117[1]
        del buf117
        buf120 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_22.run(buf120, relu_39, relu_42, buf67, buf92, 5120, 64, grid=grid(5120, 64), stream=stream0)
        del buf67
        del relu_39
        del relu_42
        buf121 = buf96; del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_16.run(buf120, buf121, 640, 512, grid=grid(640), stream=stream0)
        buf122 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_17.run(buf120, convolution_43, unsqueeze_386, buf122, 2560, 128, grid=grid(2560), stream=stream0)
        buf123 = empty((640, ), device='cuda', dtype=torch.float32)
        buf124 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf122, squeeze_130, buf123, buf124, 640, 4, grid=grid(640), stream=stream0)
        buf125 = reinterpret_tensor(buf92, (8, 640, 8, 8), (40960, 1, 5120, 640), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_18.run(buf120, convolution_43, unsqueeze_386, buf123, squeeze_130, buf121, primals_87, buf125, 512, 640, grid=grid(512, 640), stream=stream0)
        del convolution_43
        del primals_87
        del squeeze_130
        del unsqueeze_386
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf126 = aten.convolution_backward(buf125, relu_38, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_158
        buf127 = buf126[0]
        buf128 = buf126[1]
        del buf126
        buf129 = buf113; del buf113  # reuse
        buf131 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_38, buf127, convolution_42, unsqueeze_398, buf129, buf131, 7680, 128, grid=grid(7680), stream=stream0)
        buf130 = buf114; del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf129, buf130, 1920, 4, grid=grid(1920), stream=stream0)
        buf132 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf133 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf131, squeeze_127, buf132, buf133, 1920, 4, grid=grid(1920), stream=stream0)
        buf134 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_38, buf127, convolution_42, unsqueeze_398, buf132, squeeze_127, buf130, primals_85, buf134, 512, 1920, grid=grid(512, 1920), stream=stream0)
        del buf127
        del convolution_42
        del primals_85
        del relu_38
        del squeeze_127
        del unsqueeze_398
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf135 = aten.convolution_backward(buf134, relu_37, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
        del primals_157
        buf136 = buf135[0]
        buf137 = buf135[1]
        del buf135
        buf138 = buf131; del buf131  # reuse
        buf140 = buf129; del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_37, buf136, convolution_41, unsqueeze_410, buf138, buf140, 7680, 128, grid=grid(7680), stream=stream0)
        buf139 = buf132; del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf138, buf139, 1920, 4, grid=grid(1920), stream=stream0)
        buf141 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf142 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf140, squeeze_124, buf141, buf142, 1920, 4, grid=grid(1920), stream=stream0)
        buf143 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_37, buf136, convolution_41, unsqueeze_410, buf141, squeeze_124, buf139, primals_83, buf143, 512, 1920, grid=grid(512, 1920), stream=stream0)
        del buf136
        del convolution_41
        del primals_83
        del relu_37
        del squeeze_124
        del unsqueeze_410
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf144 = aten.convolution_backward(buf143, relu_36, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_156
        buf145 = buf144[0]
        buf146 = buf144[1]
        del buf144
        buf147 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_19.run(relu_36, buf120, buf145, buf147, 640, 512, grid=grid(640), stream=stream0)
        buf148 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_20.run(relu_36, buf120, buf145, convolution_40, unsqueeze_422, buf148, 2560, 128, grid=grid(2560), stream=stream0)
        buf149 = empty((640, ), device='cuda', dtype=torch.float32)
        buf151 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf148, squeeze_121, buf149, buf151, 640, 4, grid=grid(640), stream=stream0)
        buf150 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_21.run(relu_36, buf120, buf145, convolution_40, unsqueeze_422, buf149, squeeze_121, buf147, primals_81, buf150, 512, 640, grid=grid(512, 640), stream=stream0)
        del convolution_40
        del primals_81
        del squeeze_121
        del unsqueeze_422
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf152 = aten.convolution_backward(buf150, relu_35, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf150
        del primals_155
        buf153 = buf152[0]
        buf154 = buf152[1]
        del buf152
        buf155 = buf140; del buf140  # reuse
        buf157 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_35, buf153, convolution_39, unsqueeze_434, buf155, buf157, 7680, 128, grid=grid(7680), stream=stream0)
        buf156 = buf141; del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf155, buf156, 1920, 4, grid=grid(1920), stream=stream0)
        buf158 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf159 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf157, squeeze_118, buf158, buf159, 1920, 4, grid=grid(1920), stream=stream0)
        buf160 = buf143; del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_35, buf153, convolution_39, unsqueeze_434, buf158, squeeze_118, buf156, primals_79, buf160, 512, 1920, grid=grid(512, 1920), stream=stream0)
        del buf153
        del convolution_39
        del primals_79
        del relu_35
        del squeeze_118
        del unsqueeze_434
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf161 = aten.convolution_backward(buf160, relu_34, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
        del primals_154
        buf162 = buf161[0]
        buf163 = buf161[1]
        del buf161
        buf164 = buf157; del buf157  # reuse
        buf166 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_34, buf162, convolution_38, unsqueeze_446, buf164, buf166, 7680, 128, grid=grid(7680), stream=stream0)
        buf165 = buf158; del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf164, buf165, 1920, 4, grid=grid(1920), stream=stream0)
        buf167 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf168 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf166, squeeze_115, buf167, buf168, 1920, 4, grid=grid(1920), stream=stream0)
        buf169 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_34, buf162, convolution_38, unsqueeze_446, buf167, squeeze_115, buf165, primals_77, buf169, 512, 1920, grid=grid(512, 1920), stream=stream0)
        del buf162
        del convolution_38
        del primals_77
        del relu_34
        del squeeze_115
        del unsqueeze_446
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf170 = aten.convolution_backward(buf169, relu_33, primals_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_153
        buf171 = buf170[0]
        buf172 = buf170[1]
        del buf170
        buf173 = buf120; del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_23.run(buf173, relu_33, relu_36, buf145, buf171, 5120, 64, grid=grid(5120, 64), stream=stream0)
        del buf145
        del relu_33
        del relu_36
        buf174 = buf149; del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_16.run(buf173, buf174, 640, 512, grid=grid(640), stream=stream0)
        buf175 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_17.run(buf173, convolution_37, unsqueeze_458, buf175, 2560, 128, grid=grid(2560), stream=stream0)
        buf176 = empty((640, ), device='cuda', dtype=torch.float32)
        buf177 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf175, squeeze_112, buf176, buf177, 640, 4, grid=grid(640), stream=stream0)
        buf178 = reinterpret_tensor(buf171, (8, 640, 8, 8), (40960, 1, 5120, 640), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_18.run(buf173, convolution_37, unsqueeze_458, buf176, squeeze_112, buf174, primals_75, buf178, 512, 640, grid=grid(512, 640), stream=stream0)
        del convolution_37
        del primals_75
        del squeeze_112
        del unsqueeze_458
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf179 = aten.convolution_backward(buf178, relu_32, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_152
        buf180 = buf179[0]
        buf181 = buf179[1]
        del buf179
        buf182 = buf166; del buf166  # reuse
        buf184 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_32, buf180, convolution_36, unsqueeze_470, buf182, buf184, 7680, 128, grid=grid(7680), stream=stream0)
        buf183 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf182, buf183, 1920, 4, grid=grid(1920), stream=stream0)
        buf185 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf186 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf184, squeeze_109, buf185, buf186, 1920, 4, grid=grid(1920), stream=stream0)
        buf187 = buf169; del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_32, buf180, convolution_36, unsqueeze_470, buf185, squeeze_109, buf183, primals_73, buf187, 512, 1920, grid=grid(512, 1920), stream=stream0)
        del buf180
        del convolution_36
        del primals_73
        del relu_32
        del squeeze_109
        del unsqueeze_470
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf188 = aten.convolution_backward(buf187, relu_31, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
        del primals_151
        buf189 = buf188[0]
        buf190 = buf188[1]
        del buf188
        buf191 = buf184; del buf184  # reuse
        buf193 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_31, buf189, convolution_35, unsqueeze_482, buf191, buf193, 7680, 128, grid=grid(7680), stream=stream0)
        buf192 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf191, buf192, 1920, 4, grid=grid(1920), stream=stream0)
        buf194 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf195 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf193, squeeze_106, buf194, buf195, 1920, 4, grid=grid(1920), stream=stream0)
        buf196 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_31, buf189, convolution_35, unsqueeze_482, buf194, squeeze_106, buf192, primals_71, buf196, 512, 1920, grid=grid(512, 1920), stream=stream0)
        del buf189
        del convolution_35
        del primals_71
        del relu_31
        del squeeze_106
        del unsqueeze_482
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf197 = aten.convolution_backward(buf196, relu_30, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_150
        buf198 = buf197[0]
        buf199 = buf197[1]
        del buf197
        buf200 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_19.run(relu_30, buf173, buf198, buf200, 640, 512, grid=grid(640), stream=stream0)
        buf201 = buf175; del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_20.run(relu_30, buf173, buf198, convolution_34, unsqueeze_494, buf201, 2560, 128, grid=grid(2560), stream=stream0)
        buf202 = empty((640, ), device='cuda', dtype=torch.float32)
        buf204 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf201, squeeze_103, buf202, buf204, 640, 4, grid=grid(640), stream=stream0)
        buf203 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_21.run(relu_30, buf173, buf198, convolution_34, unsqueeze_494, buf202, squeeze_103, buf200, primals_69, buf203, 512, 640, grid=grid(512, 640), stream=stream0)
        del convolution_34
        del primals_69
        del squeeze_103
        del unsqueeze_494
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf205 = aten.convolution_backward(buf203, relu_29, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf203
        del primals_149
        buf206 = buf205[0]
        buf207 = buf205[1]
        del buf205
        buf208 = buf193; del buf193  # reuse
        buf210 = buf191; del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_29, buf206, convolution_33, unsqueeze_506, buf208, buf210, 7680, 128, grid=grid(7680), stream=stream0)
        buf209 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf208, buf209, 1920, 4, grid=grid(1920), stream=stream0)
        buf211 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf212 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf210, squeeze_100, buf211, buf212, 1920, 4, grid=grid(1920), stream=stream0)
        buf213 = buf196; del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_29, buf206, convolution_33, unsqueeze_506, buf211, squeeze_100, buf209, primals_67, buf213, 512, 1920, grid=grid(512, 1920), stream=stream0)
        del buf206
        del convolution_33
        del primals_67
        del relu_29
        del squeeze_100
        del unsqueeze_506
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf214 = aten.convolution_backward(buf213, relu_28, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
        del primals_148
        buf215 = buf214[0]
        buf216 = buf214[1]
        del buf214
        buf217 = buf210; del buf210  # reuse
        buf219 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_28, buf215, convolution_32, unsqueeze_518, buf217, buf219, 7680, 128, grid=grid(7680), stream=stream0)
        buf218 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf217, buf218, 1920, 4, grid=grid(1920), stream=stream0)
        buf220 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf221 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf219, squeeze_97, buf220, buf221, 1920, 4, grid=grid(1920), stream=stream0)
        buf222 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_28, buf215, convolution_32, unsqueeze_518, buf220, squeeze_97, buf218, primals_65, buf222, 512, 1920, grid=grid(512, 1920), stream=stream0)
        del buf215
        del convolution_32
        del primals_65
        del relu_28
        del squeeze_97
        del unsqueeze_518
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf223 = aten.convolution_backward(buf222, relu_27, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_147
        buf224 = buf223[0]
        buf225 = buf223[1]
        del buf223
        buf226 = buf173; del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_23.run(buf226, relu_27, relu_30, buf198, buf224, 5120, 64, grid=grid(5120, 64), stream=stream0)
        del relu_27
        del relu_30
        buf227 = buf202; del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_16.run(buf226, buf227, 640, 512, grid=grid(640), stream=stream0)
        buf228 = buf201; del buf201  # reuse
        buf235 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_24.run(buf226, convolution_31, unsqueeze_530, convolution_30, unsqueeze_542, buf228, buf235, 2560, 128, grid=grid(2560), stream=stream0)
        buf229 = empty((640, ), device='cuda', dtype=torch.float32)
        buf230 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf228, squeeze_94, buf229, buf230, 640, 4, grid=grid(640), stream=stream0)
        del buf228
        buf236 = empty((640, ), device='cuda', dtype=torch.float32)
        buf237 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf235, squeeze_91, buf236, buf237, 640, 4, grid=grid(640), stream=stream0)
        buf231 = reinterpret_tensor(buf224, (8, 640, 8, 8), (40960, 1, 5120, 640), 0); del buf224  # reuse
        buf238 = reinterpret_tensor(buf198, (8, 640, 8, 8), (40960, 1, 5120, 640), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_25.run(buf226, convolution_31, unsqueeze_530, buf229, squeeze_94, buf227, primals_63, convolution_30, unsqueeze_542, buf236, squeeze_91, primals_61, buf231, buf238, 512, 640, grid=grid(512, 640), stream=stream0)
        del buf226
        del convolution_30
        del convolution_31
        del primals_61
        del primals_63
        del squeeze_91
        del squeeze_94
        del unsqueeze_530
        del unsqueeze_542
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf232 = aten.convolution_backward(buf231, relu_24, primals_146, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf231
        del primals_146
        buf233 = buf232[0]
        buf234 = buf232[1]
        del buf232
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf239 = aten.convolution_backward(buf238, relu_26, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_145
        buf240 = buf239[0]
        buf241 = buf239[1]
        del buf239
        buf242 = buf219; del buf219  # reuse
        buf244 = buf217; del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_9.run(relu_26, buf240, convolution_29, unsqueeze_554, buf242, buf244, 7680, 128, grid=grid(7680), stream=stream0)
        buf243 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_10.run(buf242, buf243, 1920, 4, grid=grid(1920), stream=stream0)
        del buf242
        buf245 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf246 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_11.run(buf244, squeeze_88, buf245, buf246, 1920, 4, grid=grid(1920), stream=stream0)
        del buf244
        buf247 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12.run(relu_26, buf240, convolution_29, unsqueeze_554, buf245, squeeze_88, buf243, primals_59, buf247, 512, 1920, grid=grid(512, 1920), stream=stream0)
        del buf240
        del convolution_29
        del primals_59
        del relu_26
        del squeeze_88
        del unsqueeze_554
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf248 = aten.convolution_backward(buf247, relu_25, primals_144, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
        del buf247
        del primals_144
        buf249 = buf248[0]
        buf250 = buf248[1]
        del buf248
        buf251 = empty((1920, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_26.run(relu_25, buf249, buf251, 30720, 128, grid=grid(30720), stream=stream0)
        buf252 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_27.run(buf251, buf252, 1920, 16, grid=grid(1920), stream=stream0)
        buf253 = reinterpret_tensor(buf251, (1920, 16), (1, 1920), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(relu_25, buf249, convolution_28, unsqueeze_566, buf253, 30720, 128, grid=grid(30720), stream=stream0)
        buf254 = empty((1920, ), device='cuda', dtype=torch.float32)
        buf255 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_29.run(buf253, squeeze_85, buf254, buf255, 1920, 16, grid=grid(1920), stream=stream0)
        del buf253
        buf256 = empty_strided((8, 1920, 16, 16), (491520, 1, 30720, 1920), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_30.run(relu_25, buf249, convolution_28, unsqueeze_566, buf254, squeeze_85, buf252, primals_57, buf256, 2048, 1920, grid=grid(2048, 1920), stream=stream0)
        del buf249
        del buf254
        del convolution_28
        del primals_57
        del relu_25
        del squeeze_85
        del unsqueeze_566
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf257 = aten.convolution_backward(buf256, relu_24, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf256
        del primals_143
        buf258 = buf257[0]
        buf259 = buf257[1]
        del buf257
        buf260 = buf236; del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_31.run(relu_24, buf233, buf258, buf260, 640, 2048, grid=grid(640), stream=stream0)
        buf261 = reinterpret_tensor(buf5, (640, 16), (16, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_32.run(relu_24, buf233, buf258, convolution_27, unsqueeze_578, buf261, 10240, 128, grid=grid(10240), stream=stream0)
        buf262 = buf229; del buf229  # reuse
        buf264 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_33.run(buf261, squeeze_82, buf262, buf264, 640, 16, grid=grid(640), stream=stream0)
        buf263 = reinterpret_tensor(buf8, (8, 640, 16, 16), (163840, 1, 10240, 640), 0); del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_34.run(relu_24, buf233, buf258, convolution_27, unsqueeze_578, buf262, squeeze_82, buf260, primals_55, buf263, 2048, 640, grid=grid(2048, 640), stream=stream0)
        del convolution_27
        del primals_55
        del squeeze_82
        del unsqueeze_578
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf265 = aten.convolution_backward(buf263, relu_23, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf263
        del primals_142
        buf266 = buf265[0]
        buf267 = buf265[1]
        del buf265
        buf268 = reinterpret_tensor(buf235, (160, 16), (16, 1), 0); del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_35.run(relu_23, buf266, buf268, 2560, 128, grid=grid(2560), stream=stream0)
        buf269 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_36.run(buf268, buf269, 160, 16, grid=grid(160), stream=stream0)
        buf270 = reinterpret_tensor(buf268, (160, 16), (1, 160), 0); del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_23, buf266, convolution_26, unsqueeze_590, buf270, 2560, 128, grid=grid(2560), stream=stream0)
        buf271 = empty((160, ), device='cuda', dtype=torch.float32)
        buf272 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf270, squeeze_79, buf271, buf272, 160, 16, grid=grid(160), stream=stream0)
        buf273 = reinterpret_tensor(buf238, (8, 160, 16, 16), (40960, 1, 2560, 160), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39.run(relu_23, buf266, convolution_26, unsqueeze_590, buf271, squeeze_79, buf269, primals_53, buf273, 2048, 160, grid=grid(2048, 160), stream=stream0)
        del buf266
        del convolution_26
        del primals_53
        del relu_23
        del squeeze_79
        del unsqueeze_590
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf274 = aten.convolution_backward(buf273, relu_22, primals_141, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_141
        buf275 = buf274[0]
        buf276 = buf274[1]
        del buf274
        buf277 = reinterpret_tensor(buf270, (160, 16), (16, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_35.run(relu_22, buf275, buf277, 2560, 128, grid=grid(2560), stream=stream0)
        buf278 = buf271; del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_36.run(buf277, buf278, 160, 16, grid=grid(160), stream=stream0)
        buf279 = reinterpret_tensor(buf277, (160, 16), (1, 160), 0); del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_22, buf275, convolution_25, unsqueeze_602, buf279, 2560, 128, grid=grid(2560), stream=stream0)
        buf280 = empty((160, ), device='cuda', dtype=torch.float32)
        buf281 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf279, squeeze_76, buf280, buf281, 160, 16, grid=grid(160), stream=stream0)
        buf282 = buf273; del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39.run(relu_22, buf275, convolution_25, unsqueeze_602, buf280, squeeze_76, buf278, primals_51, buf282, 2048, 160, grid=grid(2048, 160), stream=stream0)
        del buf275
        del convolution_25
        del primals_51
        del relu_22
        del squeeze_76
        del unsqueeze_602
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf283 = aten.convolution_backward(buf282, relu_21, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_140
        buf284 = buf283[0]
        buf285 = buf283[1]
        del buf283
        buf286 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_40.run(buf286, relu_21, relu_24, buf258, buf284, 5120, 256, grid=grid(5120, 256), stream=stream0)
        del buf258
        del relu_21
        del relu_24
        buf287 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_41.run(buf286, buf287, 640, 2048, grid=grid(640), stream=stream0)
        buf288 = buf261; del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_42.run(buf286, convolution_24, unsqueeze_614, buf288, 10240, 128, grid=grid(10240), stream=stream0)
        buf289 = empty((640, ), device='cuda', dtype=torch.float32)
        buf290 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_33.run(buf288, squeeze_73, buf289, buf290, 640, 16, grid=grid(640), stream=stream0)
        buf291 = reinterpret_tensor(buf284, (8, 640, 16, 16), (163840, 1, 10240, 640), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_43.run(buf286, convolution_24, unsqueeze_614, buf289, squeeze_73, buf287, primals_49, buf291, 2048, 640, grid=grid(2048, 640), stream=stream0)
        del convolution_24
        del primals_49
        del squeeze_73
        del unsqueeze_614
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf292 = aten.convolution_backward(buf291, relu_20, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_139
        buf293 = buf292[0]
        buf294 = buf292[1]
        del buf292
        buf295 = reinterpret_tensor(buf279, (160, 16), (16, 1), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_35.run(relu_20, buf293, buf295, 2560, 128, grid=grid(2560), stream=stream0)
        buf296 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_36.run(buf295, buf296, 160, 16, grid=grid(160), stream=stream0)
        buf297 = reinterpret_tensor(buf295, (160, 16), (1, 160), 0); del buf295  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_20, buf293, convolution_23, unsqueeze_626, buf297, 2560, 128, grid=grid(2560), stream=stream0)
        buf298 = empty((160, ), device='cuda', dtype=torch.float32)
        buf299 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf297, squeeze_70, buf298, buf299, 160, 16, grid=grid(160), stream=stream0)
        buf300 = buf282; del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39.run(relu_20, buf293, convolution_23, unsqueeze_626, buf298, squeeze_70, buf296, primals_47, buf300, 2048, 160, grid=grid(2048, 160), stream=stream0)
        del buf293
        del convolution_23
        del primals_47
        del relu_20
        del squeeze_70
        del unsqueeze_626
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf301 = aten.convolution_backward(buf300, relu_19, primals_138, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_138
        buf302 = buf301[0]
        buf303 = buf301[1]
        del buf301
        buf304 = reinterpret_tensor(buf297, (160, 16), (16, 1), 0); del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_35.run(relu_19, buf302, buf304, 2560, 128, grid=grid(2560), stream=stream0)
        buf305 = buf298; del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_36.run(buf304, buf305, 160, 16, grid=grid(160), stream=stream0)
        buf306 = reinterpret_tensor(buf304, (160, 16), (1, 160), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_19, buf302, convolution_22, unsqueeze_638, buf306, 2560, 128, grid=grid(2560), stream=stream0)
        buf307 = empty((160, ), device='cuda', dtype=torch.float32)
        buf308 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf306, squeeze_67, buf307, buf308, 160, 16, grid=grid(160), stream=stream0)
        buf309 = buf300; del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39.run(relu_19, buf302, convolution_22, unsqueeze_638, buf307, squeeze_67, buf305, primals_45, buf309, 2048, 160, grid=grid(2048, 160), stream=stream0)
        del buf302
        del convolution_22
        del primals_45
        del relu_19
        del squeeze_67
        del unsqueeze_638
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf310 = aten.convolution_backward(buf309, relu_18, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_137
        buf311 = buf310[0]
        buf312 = buf310[1]
        del buf310
        buf313 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_31.run(relu_18, buf286, buf311, buf313, 640, 2048, grid=grid(640), stream=stream0)
        buf314 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_32.run(relu_18, buf286, buf311, convolution_21, unsqueeze_650, buf314, 10240, 128, grid=grid(10240), stream=stream0)
        buf315 = empty((640, ), device='cuda', dtype=torch.float32)
        buf317 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_33.run(buf314, squeeze_64, buf315, buf317, 640, 16, grid=grid(640), stream=stream0)
        buf316 = buf291; del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_34.run(relu_18, buf286, buf311, convolution_21, unsqueeze_650, buf315, squeeze_64, buf313, primals_43, buf316, 2048, 640, grid=grid(2048, 640), stream=stream0)
        del convolution_21
        del primals_43
        del squeeze_64
        del unsqueeze_650
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf318 = aten.convolution_backward(buf316, relu_17, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf316
        del primals_136
        buf319 = buf318[0]
        buf320 = buf318[1]
        del buf318
        buf321 = reinterpret_tensor(buf306, (160, 16), (16, 1), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_35.run(relu_17, buf319, buf321, 2560, 128, grid=grid(2560), stream=stream0)
        buf322 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_36.run(buf321, buf322, 160, 16, grid=grid(160), stream=stream0)
        buf323 = reinterpret_tensor(buf321, (160, 16), (1, 160), 0); del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_17, buf319, convolution_20, unsqueeze_662, buf323, 2560, 128, grid=grid(2560), stream=stream0)
        buf324 = empty((160, ), device='cuda', dtype=torch.float32)
        buf325 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf323, squeeze_61, buf324, buf325, 160, 16, grid=grid(160), stream=stream0)
        buf326 = buf309; del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39.run(relu_17, buf319, convolution_20, unsqueeze_662, buf324, squeeze_61, buf322, primals_41, buf326, 2048, 160, grid=grid(2048, 160), stream=stream0)
        del buf319
        del convolution_20
        del primals_41
        del relu_17
        del squeeze_61
        del unsqueeze_662
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf327 = aten.convolution_backward(buf326, relu_16, primals_135, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_135
        buf328 = buf327[0]
        buf329 = buf327[1]
        del buf327
        buf330 = reinterpret_tensor(buf323, (160, 16), (16, 1), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_35.run(relu_16, buf328, buf330, 2560, 128, grid=grid(2560), stream=stream0)
        buf331 = buf324; del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_36.run(buf330, buf331, 160, 16, grid=grid(160), stream=stream0)
        buf332 = reinterpret_tensor(buf330, (160, 16), (1, 160), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_16, buf328, convolution_19, unsqueeze_674, buf332, 2560, 128, grid=grid(2560), stream=stream0)
        buf333 = empty((160, ), device='cuda', dtype=torch.float32)
        buf334 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf332, squeeze_58, buf333, buf334, 160, 16, grid=grid(160), stream=stream0)
        buf335 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39.run(relu_16, buf328, convolution_19, unsqueeze_674, buf333, squeeze_58, buf331, primals_39, buf335, 2048, 160, grid=grid(2048, 160), stream=stream0)
        del buf328
        del convolution_19
        del primals_39
        del relu_16
        del squeeze_58
        del unsqueeze_674
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf336 = aten.convolution_backward(buf335, relu_15, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_134
        buf337 = buf336[0]
        buf338 = buf336[1]
        del buf336
        buf339 = buf286; del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_40.run(buf339, relu_15, relu_18, buf311, buf337, 5120, 256, grid=grid(5120, 256), stream=stream0)
        del buf311
        del relu_15
        del relu_18
        buf340 = buf315; del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_41.run(buf339, buf340, 640, 2048, grid=grid(640), stream=stream0)
        buf341 = buf314; del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_42.run(buf339, convolution_18, unsqueeze_686, buf341, 10240, 128, grid=grid(10240), stream=stream0)
        buf342 = empty((640, ), device='cuda', dtype=torch.float32)
        buf343 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_33.run(buf341, squeeze_55, buf342, buf343, 640, 16, grid=grid(640), stream=stream0)
        buf344 = reinterpret_tensor(buf337, (8, 640, 16, 16), (163840, 1, 10240, 640), 0); del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_43.run(buf339, convolution_18, unsqueeze_686, buf342, squeeze_55, buf340, primals_37, buf344, 2048, 640, grid=grid(2048, 640), stream=stream0)
        del convolution_18
        del primals_37
        del squeeze_55
        del unsqueeze_686
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf345 = aten.convolution_backward(buf344, relu_14, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_133
        buf346 = buf345[0]
        buf347 = buf345[1]
        del buf345
        buf348 = reinterpret_tensor(buf332, (160, 16), (16, 1), 0); del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_35.run(relu_14, buf346, buf348, 2560, 128, grid=grid(2560), stream=stream0)
        buf349 = buf333; del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_36.run(buf348, buf349, 160, 16, grid=grid(160), stream=stream0)
        buf350 = reinterpret_tensor(buf348, (160, 16), (1, 160), 0); del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_14, buf346, convolution_17, unsqueeze_698, buf350, 2560, 128, grid=grid(2560), stream=stream0)
        buf351 = empty((160, ), device='cuda', dtype=torch.float32)
        buf352 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf350, squeeze_52, buf351, buf352, 160, 16, grid=grid(160), stream=stream0)
        buf353 = buf335; del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39.run(relu_14, buf346, convolution_17, unsqueeze_698, buf351, squeeze_52, buf349, primals_35, buf353, 2048, 160, grid=grid(2048, 160), stream=stream0)
        del buf346
        del convolution_17
        del primals_35
        del relu_14
        del squeeze_52
        del unsqueeze_698
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf354 = aten.convolution_backward(buf353, relu_13, primals_132, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_132
        buf355 = buf354[0]
        buf356 = buf354[1]
        del buf354
        buf357 = reinterpret_tensor(buf350, (160, 16), (16, 1), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_35.run(relu_13, buf355, buf357, 2560, 128, grid=grid(2560), stream=stream0)
        buf358 = buf351; del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_36.run(buf357, buf358, 160, 16, grid=grid(160), stream=stream0)
        buf359 = reinterpret_tensor(buf357, (160, 16), (1, 160), 0); del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_13, buf355, convolution_16, unsqueeze_710, buf359, 2560, 128, grid=grid(2560), stream=stream0)
        buf360 = empty((160, ), device='cuda', dtype=torch.float32)
        buf361 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf359, squeeze_49, buf360, buf361, 160, 16, grid=grid(160), stream=stream0)
        buf362 = buf353; del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39.run(relu_13, buf355, convolution_16, unsqueeze_710, buf360, squeeze_49, buf358, primals_33, buf362, 2048, 160, grid=grid(2048, 160), stream=stream0)
        del buf355
        del convolution_16
        del primals_33
        del relu_13
        del squeeze_49
        del unsqueeze_710
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf363 = aten.convolution_backward(buf362, relu_12, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_131
        buf364 = buf363[0]
        buf365 = buf363[1]
        del buf363
        buf366 = buf342; del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_31.run(relu_12, buf339, buf364, buf366, 640, 2048, grid=grid(640), stream=stream0)
        buf367 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_32.run(relu_12, buf339, buf364, convolution_15, unsqueeze_722, buf367, 10240, 128, grid=grid(10240), stream=stream0)
        buf368 = empty((640, ), device='cuda', dtype=torch.float32)
        buf370 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_33.run(buf367, squeeze_46, buf368, buf370, 640, 16, grid=grid(640), stream=stream0)
        buf369 = buf344; del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_34.run(relu_12, buf339, buf364, convolution_15, unsqueeze_722, buf368, squeeze_46, buf366, primals_31, buf369, 2048, 640, grid=grid(2048, 640), stream=stream0)
        del convolution_15
        del primals_31
        del squeeze_46
        del unsqueeze_722
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf371 = aten.convolution_backward(buf369, relu_11, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf369
        del primals_130
        buf372 = buf371[0]
        buf373 = buf371[1]
        del buf371
        buf374 = reinterpret_tensor(buf359, (160, 16), (16, 1), 0); del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_35.run(relu_11, buf372, buf374, 2560, 128, grid=grid(2560), stream=stream0)
        buf375 = buf360; del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_36.run(buf374, buf375, 160, 16, grid=grid(160), stream=stream0)
        buf376 = reinterpret_tensor(buf374, (160, 16), (1, 160), 0); del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_11, buf372, convolution_14, unsqueeze_734, buf376, 2560, 128, grid=grid(2560), stream=stream0)
        buf377 = empty((160, ), device='cuda', dtype=torch.float32)
        buf378 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf376, squeeze_43, buf377, buf378, 160, 16, grid=grid(160), stream=stream0)
        buf379 = buf362; del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39.run(relu_11, buf372, convolution_14, unsqueeze_734, buf377, squeeze_43, buf375, primals_29, buf379, 2048, 160, grid=grid(2048, 160), stream=stream0)
        del buf372
        del convolution_14
        del primals_29
        del relu_11
        del squeeze_43
        del unsqueeze_734
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf380 = aten.convolution_backward(buf379, relu_10, primals_129, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_129
        buf381 = buf380[0]
        buf382 = buf380[1]
        del buf380
        buf383 = reinterpret_tensor(buf376, (160, 16), (16, 1), 0); del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_35.run(relu_10, buf381, buf383, 2560, 128, grid=grid(2560), stream=stream0)
        buf384 = buf377; del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_36.run(buf383, buf384, 160, 16, grid=grid(160), stream=stream0)
        buf385 = reinterpret_tensor(buf383, (160, 16), (1, 160), 0); del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_10, buf381, convolution_13, unsqueeze_746, buf385, 2560, 128, grid=grid(2560), stream=stream0)
        buf386 = empty((160, ), device='cuda', dtype=torch.float32)
        buf387 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf385, squeeze_40, buf386, buf387, 160, 16, grid=grid(160), stream=stream0)
        buf388 = buf379; del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39.run(relu_10, buf381, convolution_13, unsqueeze_746, buf386, squeeze_40, buf384, primals_27, buf388, 2048, 160, grid=grid(2048, 160), stream=stream0)
        del buf381
        del convolution_13
        del primals_27
        del relu_10
        del squeeze_40
        del unsqueeze_746
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf389 = aten.convolution_backward(buf388, relu_9, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_128
        buf390 = buf389[0]
        buf391 = buf389[1]
        del buf389
        buf392 = buf339; del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_40.run(buf392, relu_9, relu_12, buf364, buf390, 5120, 256, grid=grid(5120, 256), stream=stream0)
        del relu_12
        del relu_9
        buf393 = buf368; del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_41.run(buf392, buf393, 640, 2048, grid=grid(640), stream=stream0)
        buf394 = buf367; del buf367  # reuse
        buf401 = reinterpret_tensor(buf3, (640, 16), (16, 1), 0); del buf3  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_44.run(buf392, convolution_12, unsqueeze_758, convolution_11, unsqueeze_770, buf394, buf401, 10240, 128, grid=grid(10240), stream=stream0)
        buf395 = empty((640, ), device='cuda', dtype=torch.float32)
        buf396 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_33.run(buf394, squeeze_37, buf395, buf396, 640, 16, grid=grid(640), stream=stream0)
        del buf394
        buf402 = empty((640, ), device='cuda', dtype=torch.float32)
        buf403 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_33.run(buf401, squeeze_34, buf402, buf403, 640, 16, grid=grid(640), stream=stream0)
        buf397 = reinterpret_tensor(buf390, (8, 640, 16, 16), (163840, 1, 10240, 640), 0); del buf390  # reuse
        buf404 = reinterpret_tensor(buf364, (8, 640, 16, 16), (163840, 1, 10240, 640), 0); del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_45.run(buf392, convolution_12, unsqueeze_758, buf395, squeeze_37, buf393, primals_25, convolution_11, unsqueeze_770, buf402, squeeze_34, primals_23, buf397, buf404, 2048, 640, grid=grid(2048, 640), stream=stream0)
        del buf392
        del buf395
        del buf402
        del convolution_11
        del convolution_12
        del primals_23
        del primals_25
        del squeeze_34
        del squeeze_37
        del unsqueeze_758
        del unsqueeze_770
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf398 = aten.convolution_backward(buf397, relu_6, primals_127, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf397
        del primals_127
        buf399 = buf398[0]
        buf400 = buf398[1]
        del buf398
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf405 = aten.convolution_backward(buf404, relu_8, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_126
        buf406 = buf405[0]
        buf407 = buf405[1]
        del buf405
        buf408 = reinterpret_tensor(buf385, (160, 16), (16, 1), 0); del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_35.run(relu_8, buf406, buf408, 2560, 128, grid=grid(2560), stream=stream0)
        buf409 = buf386; del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_36.run(buf408, buf409, 160, 16, grid=grid(160), stream=stream0)
        buf410 = reinterpret_tensor(buf408, (160, 16), (1, 160), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_8, buf406, convolution_10, unsqueeze_782, buf410, 2560, 128, grid=grid(2560), stream=stream0)
        buf411 = empty((160, ), device='cuda', dtype=torch.float32)
        buf412 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf410, squeeze_31, buf411, buf412, 160, 16, grid=grid(160), stream=stream0)
        del buf410
        buf413 = buf388; del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39.run(relu_8, buf406, convolution_10, unsqueeze_782, buf411, squeeze_31, buf409, primals_21, buf413, 2048, 160, grid=grid(2048, 160), stream=stream0)
        del buf406
        del convolution_10
        del primals_21
        del relu_8
        del squeeze_31
        del unsqueeze_782
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf414 = aten.convolution_backward(buf413, relu_7, primals_125, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf413
        del primals_125
        buf415 = buf414[0]
        buf416 = buf414[1]
        del buf414
        buf417 = reinterpret_tensor(buf401, (160, 64), (64, 1), 0); del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_46.run(relu_7, buf415, buf417, 10240, 128, grid=grid(10240), stream=stream0)
        buf418 = buf411; del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_47.run(buf417, buf418, 160, 64, grid=grid(160), stream=stream0)
        buf419 = reinterpret_tensor(buf417, (160, 64), (1, 160), 0); del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_48.run(relu_7, buf415, convolution_9, unsqueeze_794, buf419, 10240, 128, grid=grid(10240), stream=stream0)
        buf420 = empty((160, ), device='cuda', dtype=torch.float32)
        buf421 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_49.run(buf419, squeeze_28, buf420, buf421, 160, 64, grid=grid(160), stream=stream0)
        del buf419
        buf422 = reinterpret_tensor(buf404, (8, 160, 32, 32), (163840, 1, 5120, 160), 0); del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_50.run(relu_7, buf415, convolution_9, unsqueeze_794, buf420, squeeze_28, buf418, primals_19, buf422, 8192, 160, grid=grid(8192, 160), stream=stream0)
        del buf415
        del buf420
        del convolution_9
        del primals_19
        del relu_7
        del squeeze_28
        del unsqueeze_794
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf423 = aten.convolution_backward(buf422, relu_6, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf422
        del primals_124
        buf424 = buf423[0]
        buf425 = buf423[1]
        del buf423
        buf426 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_51.run(relu_6, buf399, buf424, buf426, 192, 8192, grid=grid(192), stream=stream0)
        buf427 = empty((192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_52.run(relu_6, buf399, buf424, convolution_8, unsqueeze_806, buf427, 12288, 128, grid=grid(12288), stream=stream0)
        buf428 = empty((192, ), device='cuda', dtype=torch.float32)
        buf430 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_53.run(buf427, squeeze_25, buf428, buf430, 192, 64, grid=grid(192), stream=stream0)
        buf429 = empty_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_54.run(relu_6, buf399, buf424, convolution_8, unsqueeze_806, buf428, squeeze_25, buf426, primals_17, buf429, 8192, 192, grid=grid(8192, 192), stream=stream0)
        del convolution_8
        del primals_17
        del squeeze_25
        del unsqueeze_806
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf431 = aten.convolution_backward(buf429, relu_5, primals_123, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_123
        buf432 = buf431[0]
        buf433 = buf431[1]
        del buf431
        buf434 = buf427; del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_55.run(relu_5, buf432, buf434, 12288, 128, grid=grid(12288), stream=stream0)
        buf435 = buf428; del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_56.run(buf434, buf435, 192, 64, grid=grid(192), stream=stream0)
        buf436 = reinterpret_tensor(buf434, (192, 64), (1, 192), 0); del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_57.run(relu_5, buf432, convolution_7, unsqueeze_818, buf436, 12288, 128, grid=grid(12288), stream=stream0)
        buf437 = empty((192, ), device='cuda', dtype=torch.float32)
        buf438 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_58.run(buf436, squeeze_22, buf437, buf438, 192, 64, grid=grid(192), stream=stream0)
        buf439 = buf429; del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_59.run(relu_5, buf432, convolution_7, unsqueeze_818, buf437, squeeze_22, buf435, primals_15, buf439, 8192, 192, grid=grid(8192, 192), stream=stream0)
        del buf432
        del convolution_7
        del primals_15
        del relu_5
        del squeeze_22
        del unsqueeze_818
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf440 = aten.convolution_backward(buf439, relu_4, primals_122, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf439
        del primals_122
        buf441 = buf440[0]
        buf442 = buf440[1]
        del buf440
        buf443 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_60.run(buf443, relu_4, relu_6, buf424, buf441, 1536, 1024, grid=grid(1536, 1024), stream=stream0)
        del relu_4
        del relu_6
        buf444 = buf437; del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_61.run(buf443, buf444, 192, 8192, grid=grid(192), stream=stream0)
        buf445 = reinterpret_tensor(buf436, (192, 64), (64, 1), 0); del buf436  # reuse
        buf452 = empty((192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_62.run(buf443, convolution_6, unsqueeze_830, convolution_5, unsqueeze_842, buf445, buf452, 12288, 128, grid=grid(12288), stream=stream0)
        buf446 = empty((192, ), device='cuda', dtype=torch.float32)
        buf447 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_53.run(buf445, squeeze_19, buf446, buf447, 192, 64, grid=grid(192), stream=stream0)
        del buf445
        buf453 = empty((192, ), device='cuda', dtype=torch.float32)
        buf454 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_53.run(buf452, squeeze_16, buf453, buf454, 192, 64, grid=grid(192), stream=stream0)
        buf448 = reinterpret_tensor(buf441, (8, 192, 32, 32), (196608, 1, 6144, 192), 0); del buf441  # reuse
        buf455 = reinterpret_tensor(buf424, (8, 192, 32, 32), (196608, 1, 6144, 192), 0); del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_63.run(buf443, convolution_6, unsqueeze_830, buf446, squeeze_19, buf444, primals_13, convolution_5, unsqueeze_842, buf453, squeeze_16, primals_11, buf448, buf455, 8192, 192, grid=grid(8192, 192), stream=stream0)
        del buf443
        del convolution_5
        del convolution_6
        del primals_11
        del primals_13
        del squeeze_16
        del squeeze_19
        del unsqueeze_830
        del unsqueeze_842
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf449 = aten.convolution_backward(buf448, relu_2, primals_121, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf448
        del primals_121
        buf450 = buf449[0]
        buf451 = buf449[1]
        del buf449
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf456 = aten.convolution_backward(buf455, relu_3, primals_120, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_120
        buf457 = buf456[0]
        buf458 = buf456[1]
        del buf456
        buf459 = buf452; del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_55.run(relu_3, buf457, buf459, 12288, 128, grid=grid(12288), stream=stream0)
        buf460 = buf453; del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_56.run(buf459, buf460, 192, 64, grid=grid(192), stream=stream0)
        buf461 = reinterpret_tensor(buf459, (192, 64), (1, 192), 0); del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_57.run(relu_3, buf457, convolution_4, unsqueeze_854, buf461, 12288, 128, grid=grid(12288), stream=stream0)
        buf462 = buf446; del buf446  # reuse
        buf463 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_58.run(buf461, squeeze_13, buf462, buf463, 192, 64, grid=grid(192), stream=stream0)
        del buf461
        buf464 = buf455; del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_59.run(relu_3, buf457, convolution_4, unsqueeze_854, buf462, squeeze_13, buf460, primals_9, buf464, 8192, 192, grid=grid(8192, 192), stream=stream0)
        del buf457
        del buf462
        del convolution_4
        del primals_9
        del relu_3
        del squeeze_13
        del unsqueeze_854
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf465 = aten.convolution_backward(buf464, relu_2, primals_119, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf464
        del primals_119
        buf466 = buf465[0]
        buf467 = buf465[1]
        del buf465
        buf468 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_64.run(relu_2, buf450, buf466, buf468, 512, 8192, grid=grid(512), stream=stream0)
        buf469 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_65.run(buf468, buf469, 128, 4, grid=grid(128), stream=stream0)
        buf470 = empty((128, 256), device='cuda', dtype=torch.float32)
        buf477 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_66.run(relu_2, buf450, buf466, convolution_3, unsqueeze_866, convolution_2, unsqueeze_878, buf470, buf477, 32768, 128, grid=grid(32768), stream=stream0)
        buf471 = empty((128, ), device='cuda', dtype=torch.float32)
        buf473 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_67.run(buf470, squeeze_10, buf471, buf473, 128, 256, grid=grid(128), stream=stream0)
        del buf470
        buf478 = empty((128, ), device='cuda', dtype=torch.float32)
        buf480 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_67.run(buf477, squeeze_7, buf478, buf480, 128, 256, grid=grid(128), stream=stream0)
        buf472 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda', dtype=torch.float32)
        buf479 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_68.run(relu_2, buf450, buf466, convolution_3, unsqueeze_866, buf471, squeeze_10, buf469, primals_7, convolution_2, unsqueeze_878, buf478, squeeze_7, primals_5, buf472, buf479, 32768, 128, grid=grid(32768, 128), stream=stream0)
        del buf450
        del buf466
        del convolution_2
        del convolution_3
        del primals_5
        del primals_7
        del relu_2
        del squeeze_10
        del squeeze_7
        del unsqueeze_866
        del unsqueeze_878
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf474 = aten.convolution_backward(buf472, relu, primals_118, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf472
        del primals_118
        buf475 = buf474[0]
        buf476 = buf474[1]
        del buf474
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf481 = aten.convolution_backward(buf479, relu_1, primals_117, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_117
        buf482 = buf481[0]
        buf483 = buf481[1]
        del buf481
        buf484 = buf477; del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_69.run(relu_1, buf482, buf484, 32768, 128, grid=grid(32768), stream=stream0)
        buf485 = buf478; del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_70.run(buf484, buf485, 128, 256, grid=grid(128), stream=stream0)
        buf486 = reinterpret_tensor(buf484, (128, 256), (1, 128), 0); del buf484  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_71.run(relu_1, buf482, convolution_1, unsqueeze_890, buf486, 32768, 128, grid=grid(32768), stream=stream0)
        buf487 = buf471; del buf471  # reuse
        buf488 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_72.run(buf486, squeeze_4, buf487, buf488, 128, 256, grid=grid(128), stream=stream0)
        buf489 = buf479; del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_73.run(relu_1, buf482, convolution_1, unsqueeze_890, buf487, squeeze_4, buf485, primals_3, buf489, 32768, 128, grid=grid(32768, 128), stream=stream0)
        del buf482
        del buf487
        del convolution_1
        del primals_3
        del relu_1
        del squeeze_4
        del unsqueeze_890
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf490 = aten.convolution_backward(buf489, relu, primals_116, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_116
        buf491 = buf490[0]
        buf492 = buf490[1]
        del buf490
        buf493 = reinterpret_tensor(buf468, (32, 16), (16, 1), 0); del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_74.run(relu, buf475, buf491, buf493, 512, 8192, grid=grid(512), stream=stream0)
        buf494 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_75.run(buf493, buf494, 32, 16, grid=grid(32), stream=stream0)
        del buf493
        buf495 = reinterpret_tensor(buf486, (32, 1024), (1024, 1), 0); del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_76.run(relu, buf475, buf491, convolution, unsqueeze_902, buf495, 32768, 128, grid=grid(32768), stream=stream0)
        buf496 = empty((32, ), device='cuda', dtype=torch.float32)
        buf498 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_77.run(buf495, squeeze_1, buf496, buf498, 32, 1024, grid=grid(32), stream=stream0)
        del buf495
        buf497 = reinterpret_tensor(buf489, (8, 32, 128, 128), (524288, 1, 4096, 32), 0); del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_78.run(relu, buf475, buf491, convolution, unsqueeze_902, buf496, squeeze_1, buf494, primals_1, buf497, 131072, 32, grid=grid(131072, 32), stream=stream0)
        del buf475
        del buf491
        del buf496
        del convolution
        del primals_1
        del relu
        del squeeze_1
        del unsqueeze_902
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf499 = aten.convolution_backward(buf497, primals_345, primals_115, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf497
        del primals_115
        del primals_345
        buf500 = buf499[1]
        return (buf498, buf494, buf488, buf485, buf480, buf469, buf473, buf469, buf463, buf460, buf454, buf444, buf447, buf444, buf438, buf435, buf430, buf426, buf421, buf418, buf412, buf409, buf403, buf393, buf396, buf393, buf387, buf384, buf378, buf375, buf370, buf366, buf361, buf358, buf352, buf349, buf343, buf340, buf334, buf331, buf325, buf322, buf317, buf313, buf308, buf305, buf299, buf296, buf290, buf287, buf281, buf278, buf272, buf269, buf264, buf260, buf255, buf252, buf246, buf243, buf237, buf227, buf230, buf227, buf221, buf218, buf212, buf209, buf204, buf200, buf195, buf192, buf186, buf183, buf177, buf174, buf168, buf165, buf159, buf156, buf151, buf147, buf142, buf139, buf133, buf130, buf124, buf121, buf115, buf112, buf106, buf103, buf98, buf94, buf89, buf86, buf80, buf77, buf71, buf68, buf62, buf59, buf53, buf50, buf44, buf40, buf34, buf31, buf25, buf22, buf16, buf13, buf7, buf4, buf500, buf492, buf483, buf476, buf467, buf458, buf451, buf442, buf433, buf425, buf416, buf407, buf400, buf391, buf382, buf373, buf365, buf356, buf347, buf338, buf329, buf320, buf312, buf303, buf294, buf285, buf276, buf267, buf259, buf250, buf241, buf234, buf225, buf216, buf207, buf199, buf190, buf181, buf172, buf163, buf154, buf146, buf137, buf128, buf119, buf110, buf101, buf93, buf84, buf75, buf66, buf57, buf48, buf38, buf29, buf20, buf11, reinterpret_tensor(buf1, (1000, 2560), (2560, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((192, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((192, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((160, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((640, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((640, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((2560, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 160, 32, 32), (163840, 1, 5120, 160), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((8, 160, 32, 32), (163840, 1, 5120, 160), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_21 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_23 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_24 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 1920, 16, 16), (491520, 1, 30720, 1920), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_25 = rand_strided((8, 1920, 16, 16), (491520, 1, 30720, 1920), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_26 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_27 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_28 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_29 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_31 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_32 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_33 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_34 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_35 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_36 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_37 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_38 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_39 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_40 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_41 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_42 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_43 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_44 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_45 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_46 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_47 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_48 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    squeeze_160 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_49 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    squeeze_163 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_50 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    squeeze_166 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_51 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 2560, 8, 8), (163840, 1, 20480, 2560), device='cuda:0', dtype=torch.float32)
    squeeze_169 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((8, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((8, 2560, 8, 8), (163840, 1, 20480, 2560), device='cuda:0', dtype=torch.bool)
    unsqueeze_230 = rand_strided((1, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_242 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_254 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_266 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_278 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_290 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_302 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_314 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_326 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_338 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_350 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_362 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_374 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_386 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_398 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_410 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_422 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_434 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_446 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_458 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_470 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_482 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_494 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_506 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_518 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_530 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_542 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_554 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_566 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_578 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_590 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_602 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_614 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_626 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_638 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_650 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_662 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_674 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_686 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_698 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_710 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_722 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_734 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_746 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_758 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_770 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_782 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_794 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_806 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_818 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_830 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_842 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_854 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_866 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_878 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_890 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_902 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_345, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, convolution_3, squeeze_10, relu_2, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_4, convolution_7, squeeze_22, relu_5, convolution_8, squeeze_25, relu_6, convolution_9, squeeze_28, relu_7, convolution_10, squeeze_31, relu_8, convolution_11, squeeze_34, convolution_12, squeeze_37, relu_9, convolution_13, squeeze_40, relu_10, convolution_14, squeeze_43, relu_11, convolution_15, squeeze_46, relu_12, convolution_16, squeeze_49, relu_13, convolution_17, squeeze_52, relu_14, convolution_18, squeeze_55, relu_15, convolution_19, squeeze_58, relu_16, convolution_20, squeeze_61, relu_17, convolution_21, squeeze_64, relu_18, convolution_22, squeeze_67, relu_19, convolution_23, squeeze_70, relu_20, convolution_24, squeeze_73, relu_21, convolution_25, squeeze_76, relu_22, convolution_26, squeeze_79, relu_23, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, convolution_31, squeeze_94, relu_27, convolution_32, squeeze_97, relu_28, convolution_33, squeeze_100, relu_29, convolution_34, squeeze_103, relu_30, convolution_35, squeeze_106, relu_31, convolution_36, squeeze_109, relu_32, convolution_37, squeeze_112, relu_33, convolution_38, squeeze_115, relu_34, convolution_39, squeeze_118, relu_35, convolution_40, squeeze_121, relu_36, convolution_41, squeeze_124, relu_37, convolution_42, squeeze_127, relu_38, convolution_43, squeeze_130, relu_39, convolution_44, squeeze_133, relu_40, convolution_45, squeeze_136, relu_41, convolution_46, squeeze_139, relu_42, convolution_47, squeeze_142, relu_43, convolution_48, squeeze_145, relu_44, convolution_49, squeeze_148, relu_45, convolution_50, squeeze_151, relu_46, convolution_51, squeeze_154, relu_47, convolution_52, squeeze_157, relu_48, convolution_53, squeeze_160, relu_49, convolution_54, squeeze_163, relu_50, convolution_55, squeeze_166, relu_51, convolution_56, squeeze_169, clone, permute_1, le, unsqueeze_230, unsqueeze_242, unsqueeze_254, unsqueeze_266, unsqueeze_278, unsqueeze_290, unsqueeze_302, unsqueeze_314, unsqueeze_326, unsqueeze_338, unsqueeze_350, unsqueeze_362, unsqueeze_374, unsqueeze_386, unsqueeze_398, unsqueeze_410, unsqueeze_422, unsqueeze_434, unsqueeze_446, unsqueeze_458, unsqueeze_470, unsqueeze_482, unsqueeze_494, unsqueeze_506, unsqueeze_518, unsqueeze_530, unsqueeze_542, unsqueeze_554, unsqueeze_566, unsqueeze_578, unsqueeze_590, unsqueeze_602, unsqueeze_614, unsqueeze_626, unsqueeze_638, unsqueeze_650, unsqueeze_662, unsqueeze_674, unsqueeze_686, unsqueeze_698, unsqueeze_710, unsqueeze_722, unsqueeze_734, unsqueeze_746, unsqueeze_758, unsqueeze_770, unsqueeze_782, unsqueeze_794, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gernet_l', benchmark_compiled_module)
