
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


# kernel path: /tmp/torchinductor_youkaichao/r7/cr7gs4gqdrfdeya75eszv5gs5nl6235utfbes2m5rbbto24se6xr.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_hardtanh_backward_native_batch_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
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


# kernel path: /tmp/torchinductor_youkaichao/nn/cnn27ibjvqxnzuf6hnietdimovh3kejvzncmz6awgtrc2whqhss3.py
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
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_hardtanh_backward_native_batch_norm_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/hs/chsweex2gqxndj5l5ux2sqhga345p6fen5ahwaknfclruq2fihqi.py
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
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_hardtanh_backward_native_batch_norm_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kp/ckpo7qobdjzx6zn73nwdi6sirvqzzc6m4eflruk25fuasos6nicg.py
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
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_hardtanh_backward_native_batch_norm_backward_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1280
    x2 = (xindex // 62720)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (1280*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.002551020408163265
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


# kernel path: /tmp/torchinductor_youkaichao/bh/cbh65jvkpr3i6xzkypxelt6e4o7o42adldkg7hruqrhw2venzyh2.py
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
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 320
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (15680*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q4/cq4u57nryemc6c7yak7qy5fhezndf2n6jyrtdtf5npepjudhxzzc.py
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1280
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
        tmp1 = tl.load(in_ptr1 + (x0 + (320*r2) + (31360*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ti/ctiwbr2m3imokwcz5a23y7rvuopphnzixxt35gu7bqpgokgzyxub.py
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
    size_hints=[512, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4d/c4dsc6ugydxic5jcvk2d5yt2qzqpaoanjys6usnhs4p3bhmrsvcm.py
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
    size_hints=[4096, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (320*x2) + (15680*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/nd/cndb24xfq73a6tjbh2kjymf3gluc2krodgqnl7wpjmpx6dm47ebj.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_9', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/rk/crk3ftf72urokodxq4dfk5spabcyy3k4abieres2x5pcyml42nqe.py
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
    size_hints=[1024, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_10', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zl/czlf63ttgglxpacxuhyqbihu7n7mcret7tw6bl24fczaszvl2o5u.py
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
    size_hints=[1024, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_11', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/5u/c5ugxkbqkynaagrkbykl7uot7gudakblq75mbgpmy7fvwhvbgn4m.py
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
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_12', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (x2 + (960*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (49*x2) + (47040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (960*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (960*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6o/c6oea7qbch7t5m262ezr75gp6uahdzpclk4mph6iqcjiq2dijvkg.py
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
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_13', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ps/cpscysqg36c73qowg527ttsyjyhp5xqgbj7csit6d7qpydlj46v5.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_14', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/r3/cr3sedqutjt5eqnqyfbeh46qitvpafstz4dsn73cikunwkc3gdzq.py
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
    size_hints=[256, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_15', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/v6/cv636nfkpaqarelylgmh7t4dwhr4s6ieu2sar2jneqgoov2b4rmk.py
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
    size_hints=[2048, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_16', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ww/cwwbnhqh6rx47sriflz2datro3sbkitosv7zfnqztwqi6s7s5ch4.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_17', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/z7/cz7lk6sadmbzujd3i6ofbmimhwopvjcpqfisectapur73eeo5ndd.py
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
    size_hints=[2048, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_18', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/zf/czfchmsx7ukngeuxfwruacdr6njrpfqjsyfsatpbjza67v2lgoyh.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_19', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/za/czaawpspuln7hhww2zcv2fjkejii2rd2wf35sfujl62p2rb6xhgh.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_20', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/bn/cbnxu2id4k4nwga2fjnumgviw6jrplbufwwz7nrypc2kcmcgikts.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2304
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


# kernel path: /tmp/torchinductor_youkaichao/3z/c3ztwjqwbxyhkwpewvyy3sds2clanibyhuoup6fw7cwxuy7tk6zx.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/u7/cu76krpp5fwpqutyyx3f3lhfdljfxo2llugahrbvpuxnstnx2f5n.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qx/cqxrt4gmuuw5f473kth3lxsryces7zbaopzacmfr7d3ttkjtmfa5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
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
    tmp4 = tl.load(in_ptr2 + (x2 + (576*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (576*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s5/cs5w637uyb2ku2ybr3xmyqlpxt2idmkgx426uervpu74ha4gvtvo.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7488
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
        tmp3 = tl.load(in_ptr0 + (x1 + (576*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((196*x1) + (112896*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/av/cavj5kj4rscmketh2fwjgqeknphhclc4xdzwh7meagpfkxpgqrgo.py
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
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 576
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


# kernel path: /tmp/torchinductor_youkaichao/ez/cezl4dltz6aqpsilnbavuqhdlfg4d3kokldtdbkepcyaussqr5ef.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7488
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 576)
    x0 = xindex % 576
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (576*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((196*x0) + (112896*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.load(in_ptr2 + (x0 + (576*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/id/cid2p7w5gt6tdrekvvwlx4hm63w5uid5bbztpihwrc23fvkizyh2.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ls/clso5jajhyf75awhxnvylkahjoj27fe4qt5lpwqaxxeqlyzf6wtl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
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
    tmp4 = tl.load(in_ptr2 + (x2 + (576*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (576*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nu/cnu736pnutqo77zt77af6tabnfb32yoakevjyzigfzkeohtrvl4u.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (18816*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qo/cqoumf7tqrdg3f4mucqte3a34hxyvlst6i6zj5kppnqmcl7w7qiw.py
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1248
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (18816*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (96*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ah/cahpvnmbuvsfkfl643w5zmwp7ubvtcxqumfzad5ymvs6tliijyll.py
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
    size_hints=[128, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
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


# kernel path: /tmp/torchinductor_youkaichao/tk/ctklnzhdr64exjglcat327scerber4kfjfmanaihna7y67rokbmt.py
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
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (96*x2) + (18816*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/46/c46gkgsmhy2phnocevmez6kip3rv3l2gc3pjjhmxggxqwvfeidpv.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (18816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (18816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (96*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ez/cezctpb53qdsl2jxk5x3ui4dke52mznekzmtnk36iptemwmzzsyq.py
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
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (96*x2) + (18816*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/o2/co2ydcqxok433bp5jcjjtgnsf3yni7bxqp6kh3wc2faqdxyxt6jn.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (18816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (18816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (196*x0) + (18816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (96*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/vv/cvvxy6wnsv4v2wvllhinux35gvrumeg7xfld2wp3wsk2yrbwkcx6.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + (96*x2) + (18816*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e4/ce4zaqsn2habj65sczxko3my7g5qljfod2q3svoyufxbdo2plwk4.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
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
        tmp3 = tl.load(in_ptr0 + (x1 + (384*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((196*x1) + (75264*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/tx/ctxl2albbo4iqiqu3j2fviuqthhzka4f2k2dnpazshuokdpt2mov.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
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


# kernel path: /tmp/torchinductor_youkaichao/lw/clwbxe4ygkzshxrpvgrlxenxd4mykyozgj7yumsuhscfviqycygw.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((196*x0) + (75264*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.load(in_ptr2 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/6t/c6tpxuc66bji2aida6umiuwtdzyuctezybamq3kcbpnuww7izly4.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4u/c4uqta4sgv3pxgut72wlzsq7fmnwpssrecxnqy46x73ouj5euo2u.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
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
    tmp4 = tl.load(in_ptr2 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2h/c2hmnbbjdbegf5od3fpqetvc3s5pr7n2wtrcqztolrcrih6ndg6y.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zt/cztzs7nvnl7hpv6jtiv5ttjvzqbfj5v3j52easbzo6l5a6kwvbxs.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (12544*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (64*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ia/ciatmqvfl3azq7jnja76aruywumq6pcrxr6gmgv5pwlfazxapcdi.py
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
    size_hints=[64, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_45', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e6/ce6kiwdiikrsjz6og4a4diq3w2pwd3phnwaqwz5yfg5heyrr6x7i.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (64*x2) + (12544*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/ve/cve6q3anvk6s52nt2htaa5jdjeuy6mapcbzd6uot547vunzzm6za.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (64*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/of/cofjg2mqhckvm5jxbcn5ctgni74wc4wz2x57h6nh3xzjziajnmjx.py
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
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (64*x2) + (12544*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/f5/cf5vzkftu5odtyfshupjg5wn7vk6q555q564x3ew4vcxuvivvjx5.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (196*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (64*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/m4/cm4dcju6a3nthtzhy6uoqobywzxbsttidczu2oizyc2uobobnfxf.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (64*x2) + (12544*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/xi/cxibooxf5ttk7bjmqqguxcc3akmpng33jrducfop47cflg3fg3fb.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (196*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r1 + (196*x0) + (12544*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr4 + (x0 + (64*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/zn/cznlxkq2oq6zcmvvjeobobioc5jalic6l5evol3etnfd7w4im5jc.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_52', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 196
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (64*x2) + (12544*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/xf/cxfli27dorywuhekmbqinkj7k2x6bm277sy2zyuozxrjdgm5yzk3.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_53', 'mutated_arg_names': []}
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
        tmp4 = tl.load(in_ptr1 + ((196*x1) + (37632*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/hs/chss5r4cjitskebqbqoxj7ab45w3jx72dclr4766zxxgihs4cxcx.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_54', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zl/czlpwmoowyyi435o6u4iyqq6c5xgrut2j2umxlywu4zoozdtpsqd.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_55', 'mutated_arg_names': []}
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
        tmp4 = tl.load(in_ptr1 + ((196*x0) + (37632*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ao/caocwv6xxugtrbtpkf4osmf7f4keurg6s5zsawkse65kdhuleu3w.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_56', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/y5/cy5xv4uwnwy2ru2t26cwst5tm6u3vyqto4gut2zyyuukxhe4yqdx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_57', 'mutated_arg_names': []},
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
    tmp1 = tl.load(in_ptr1 + (y0 + (196*x2) + (37632*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/6v/c6vfqcj536hczmfswmggomxmohy7v2bzoz6l2ezfzikllyzqai57.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
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
        tmp0 = tl.load(in_ptr0 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((784*x1) + (150528*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tg/ctgchf2rvdkudogp6ibayxgz6npeyqony55toiptlbezgmdd4v5q.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
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


# kernel path: /tmp/torchinductor_youkaichao/76/c76a3sxgzh77k4wwkepok3bs4wuuoz3v7xkmtnswjjk4b6egcjee.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((784*x0) + (150528*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/5d/c5dumpgyzuytzfuqqy6h7uqkiweizxy4p2f5c75bx6umk7xogxyb.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 49
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


# kernel path: /tmp/torchinductor_youkaichao/b3/cb3e7xbvafgltbspoel65rr5ty7acx3jnheouc5mm6w6d5xuawea.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
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
    tmp4 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tt/ctthlzvz2jg4ysfkvkdqc5h44kdvh2vxea6n7tm4xq6gaicd3d7d.py
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
    size_hints=[32, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yb/cybm3dnifzfdjwxpsttmth5olmtu5ztkv7rr7ewsfgn4omg7uwdp.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
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
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (25088*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (32*r2) + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmrlh4xssj7ydksvym7uzuqos7yysiidarvkuzncs6vj3y5u2ox.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_65', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xl/cxleogqshjigtyeovzjpqansvzmfjoliwarli5uyeqnj4zitk3ry.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (32*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/jl/cjl4uexpwqbsoibfnpystt6nkd6tmxkydyyjrr2c5dogofu6pmg2.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (784*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (32*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/kg/ckgsihn2rvhwynduvmlmdooluqzhhslbbcphl7jtdod4clthhtli.py
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
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (32*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/g5/cg5eupcoznoc2eaml32rcdfv7upuafiltrijbyrtgouuudtuupvm.py
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
    size_hints=[32, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (784*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (784*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (32*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/cd/ccdcxeecc5dmcfjmburuyaxb3y7ueolt7ifzurlzrqi5ciymkh6m.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_70', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + (32*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/6a/c6awdrwx3ksnbfbxrdecoxtqrxuylbfuzmwp73x2k5wluhf6kkh5.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7056
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
        tmp0 = tl.load(in_ptr0 + (x1 + (144*r2) + (18432*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((784*x1) + (112896*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xy/cxygbhwz3so37vqk6ydky5gbs6eb5hwxu3glhp5qwmmfgdgpaehh.py
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
    size_hints=[256, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 144
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


# kernel path: /tmp/torchinductor_youkaichao/f5/cf573kwaudp7dph3p7c2vjur7mgags4352gty344bscj75npkikr.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7056
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
        tmp1 = tl.load(in_ptr1 + ((784*x0) + (112896*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/i6/ci6dqjc66creivetqm4gaugdvlwv5blbwphwzpgdiohgslze65pi.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 49
    RBLOCK: tl.constexpr = 64
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
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmc45qfivcym7m3z2nmkfsjqb7qxn25ieldsl5jyio2mrf745xn.py
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
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
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
    tmp4 = tl.load(in_ptr2 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (144*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/36/c36vv2z4na6f73326lj5van5oxa3qnpne4yjob6k5omzf7swbr4c.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 28224
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


# kernel path: /tmp/torchinductor_youkaichao/6f/c6fymvon2tklprbxdjzsr3ialibazs4oyvpaw3ya6si3srxpudkb.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 144
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


# kernel path: /tmp/torchinductor_youkaichao/fa/cfa4qvyxiu5tizzxzh4lketfjyg2molggxopmkxgnkn3g2y6chqr.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 28224
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
        tmp4 = tl.load(in_ptr2 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/vm/cvmhrm7lt5xsnvamiby25gyyrz3o3fos7irsgew54ndfkau4j6jo.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_79', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 144
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
        tmp0 = tl.load(in_ptr0 + (x0 + (144*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kg/ckgcimcooy5csd54ieajkcz57wdyfkn5sg24ce4eo5gxnbbbojyf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_80 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
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
    tmp4 = tl.load(in_ptr2 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (144*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fu/cfuonazcurnhv23r524bv7mfngg2ujyxfriwix4dwobdxz5eljs6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_81', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/yc/cyclyblusckoymkh2wy7cpct5gx4bfbcvmxchqxvv7y4w2dmrehj.py
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
    size_hints=[32, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_82', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/kf/ckfjufe2q74qi73spgj4mjuigmizmjd77zsrdrgdjctmktcfhqby.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_83', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/q5/cq5g6gryzoafi76eow5fxonlqh2ma7h4ttmfdisukjdf6uaj4wro.py
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
    size_hints=[32, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_84', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/6u/c6utzz7ffiehxtoiv74sya3woc5pwkxqmkib2piw5vrsj7rl7pg5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_85', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/tb/ctbtke3z3v4lmkfruonlrybpmdynb7mxfojuhmognuznc4vprwbh.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_86', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxlecywds5q3zk3gcp3wrprroarrmcrifg7aj4oz4uaakubt7eg.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_87', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/46/c46ijjd6n5zqhhlkoggfa7ea2tgdbaaqlvloc4pawximflu3sid6.py
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
    size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_88', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/zx/czxcs37iemjj3tzbmwfpu253fpvelt7vm55tme6gbuxneito3bb2.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18816
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


# kernel path: /tmp/torchinductor_youkaichao/kw/ckwmlgdvjh7fxgu6qs3eiznmzqdofkhv3s6ywypup2pujvq7gvj3.py
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
    size_hints=[128, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
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


# kernel path: /tmp/torchinductor_youkaichao/75/c75xbtujwdnwu6xmd6mmbsae3ktdbagwfv7beqfqq5ndbblwv6qf.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18816
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
        tmp4 = tl.load(in_ptr2 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/nj/cnjf5fgklitqhzyfixglg2ejv7d7gsnz52tgrotwy2tyoa6ch3lw.py
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
    size_hints=[128, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_92', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
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
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q6/cq6c45xh6pl2ielgobye7574xmy2nxdugfqhlcoegnomrvbcpya3.py
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
    size_hints=[32768, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
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
    tmp4 = tl.load(in_ptr2 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (96*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xj/cxja7oc5omcfb33y75wybtjgqces6oe5jshmfrhat4zz5ikzpqpa.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_94 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_94', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 75264
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
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


# kernel path: /tmp/torchinductor_youkaichao/hf/chf4sdg2bzauqmpyk5cqx75izqxbx6yg4xymap4aezozrt4gzfvy.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_95 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 96
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


# kernel path: /tmp/torchinductor_youkaichao/xt/cxto6r737xfbkz2lww6y4yteiervfuzz3on6hbugo6riwyjbafxl.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_96 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_96', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 75264
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
        tmp4 = tl.load(in_ptr2 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4n/c4n52fxircprxethsakoqj72fdvtwhtzrnz4iy5my6elgy5afhir.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_97 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
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
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fe/cfefnbsdx4nxvk56mj6uvxxqk6pgxdtzgygtc7vvba42l4m5abj5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_98 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_98', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100352
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
    tmp4 = tl.load(in_ptr2 + (x2 + (96*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 9.964923469387754e-06
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x2 + (96*y3)), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gn/cgnaj4yxrugiwqnnmkafflzky5pgnxwhdpqtnl44sx3q7qsrzfpa.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_99', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/67/c67jc52tua2gwy4ff3hsieknxyy2uugrxgfkudllg6nl343c7cbn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_100 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_100', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/dh/cdhhvo2d7malhqomladffkjtf4ynfdrsa6zz72lvom2sgknhyyrp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_101 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_101', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/mz/cmz66hjrzu4dol52csf6srhg62xhpsgmcc6i324nlrsv6icfnuh6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_102 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_102', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/yr/cyrh7novwdusz62mthyehtx5ggef2wvlbum7xre5vhg43rbsajih.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_103 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_103', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (16*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (12544*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3s/c3s3g6fhcq7xmbq3y5xtrcn7rhu5qp5zchb4ycu5netjlzy3x5y4.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_104 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_104', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
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


# kernel path: /tmp/torchinductor_youkaichao/qi/cqilv47pmai77vm2htkopsv26jamq4vczz6gpuivwmro3nu5dcyy.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_105 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_105', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 32
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


# kernel path: /tmp/torchinductor_youkaichao/2d/c2dkhxczk7rxqg6msjdcrjpoqiqlfgsn5suybwe3fnom2kbp65h7.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_106 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_106', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
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
        tmp4 = tl.load(in_ptr2 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/kr/ckruxiyeoqonseherqq6milyow73un72tfm3euworwoqyowxkkmt.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_107 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_107', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
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


# kernel path: /tmp/torchinductor_youkaichao/i6/ci6hdhcdoojpp4ypkaslmarkqymemsfne4ls4xqsdukxoe2jdzab.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_108 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_108', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100352
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
    tmp4 = tl.load(in_ptr2 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 9.964923469387754e-06
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x2 + (32*y3)), tmp20, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_315, convolution, squeeze_1, clamp_max, convolution_1, squeeze_4, clamp_max_1, convolution_2, squeeze_7, add_14, convolution_3, squeeze_10, clamp_max_2, convolution_4, squeeze_13, clamp_max_3, convolution_5, squeeze_16, add_29, convolution_6, squeeze_19, clamp_max_4, convolution_7, squeeze_22, clamp_max_5, convolution_8, squeeze_25, add_45, convolution_9, squeeze_28, clamp_max_6, convolution_10, squeeze_31, clamp_max_7, convolution_11, squeeze_34, add_60, convolution_12, squeeze_37, clamp_max_8, convolution_13, squeeze_40, clamp_max_9, convolution_14, squeeze_43, add_76, convolution_15, squeeze_46, clamp_max_10, convolution_16, squeeze_49, clamp_max_11, convolution_17, squeeze_52, add_92, convolution_18, squeeze_55, clamp_max_12, convolution_19, squeeze_58, clamp_max_13, convolution_20, squeeze_61, add_107, convolution_21, squeeze_64, clamp_max_14, convolution_22, squeeze_67, clamp_max_15, convolution_23, squeeze_70, add_123, convolution_24, squeeze_73, clamp_max_16, convolution_25, squeeze_76, clamp_max_17, convolution_26, squeeze_79, add_139, convolution_27, squeeze_82, clamp_max_18, convolution_28, squeeze_85, clamp_max_19, convolution_29, squeeze_88, add_155, convolution_30, squeeze_91, clamp_max_20, convolution_31, squeeze_94, clamp_max_21, convolution_32, squeeze_97, add_170, convolution_33, squeeze_100, clamp_max_22, convolution_34, squeeze_103, clamp_max_23, convolution_35, squeeze_106, add_186, convolution_36, squeeze_109, clamp_max_24, convolution_37, squeeze_112, clamp_max_25, convolution_38, squeeze_115, add_202, convolution_39, squeeze_118, clamp_max_26, convolution_40, squeeze_121, clamp_max_27, convolution_41, squeeze_124, add_217, convolution_42, squeeze_127, clamp_max_28, convolution_43, squeeze_130, clamp_max_29, convolution_44, squeeze_133, add_233, convolution_45, squeeze_136, clamp_max_30, convolution_46, squeeze_139, clamp_max_31, convolution_47, squeeze_142, add_249, convolution_48, squeeze_145, clamp_max_32, convolution_49, squeeze_148, clamp_max_33, convolution_50, squeeze_151, add_264, convolution_51, squeeze_154, view, permute_1, bitwise_or, unsqueeze_210, unsqueeze_222, bitwise_or_1, unsqueeze_234, bitwise_or_2, unsqueeze_246, unsqueeze_258, bitwise_or_3, unsqueeze_270, bitwise_or_4, unsqueeze_282, unsqueeze_294, bitwise_or_5, unsqueeze_306, bitwise_or_6, unsqueeze_318, unsqueeze_330, bitwise_or_7, unsqueeze_342, bitwise_or_8, unsqueeze_354, unsqueeze_366, bitwise_or_9, unsqueeze_378, bitwise_or_10, unsqueeze_390, unsqueeze_402, bitwise_or_11, unsqueeze_414, bitwise_or_12, unsqueeze_426, unsqueeze_438, bitwise_or_13, unsqueeze_450, bitwise_or_14, unsqueeze_462, unsqueeze_474, bitwise_or_15, unsqueeze_486, bitwise_or_16, unsqueeze_498, unsqueeze_510, bitwise_or_17, unsqueeze_522, bitwise_or_18, unsqueeze_534, unsqueeze_546, bitwise_or_19, unsqueeze_558, bitwise_or_20, unsqueeze_570, unsqueeze_582, bitwise_or_21, unsqueeze_594, bitwise_or_22, unsqueeze_606, unsqueeze_618, bitwise_or_23, unsqueeze_630, bitwise_or_24, unsqueeze_642, unsqueeze_654, bitwise_or_25, unsqueeze_666, bitwise_or_26, unsqueeze_678, unsqueeze_690, bitwise_or_27, unsqueeze_702, bitwise_or_28, unsqueeze_714, unsqueeze_726, bitwise_or_29, unsqueeze_738, bitwise_or_30, unsqueeze_750, unsqueeze_762, bitwise_or_31, unsqueeze_774, bitwise_or_32, unsqueeze_786, unsqueeze_798, bitwise_or_33, unsqueeze_810, bitwise_or_34, unsqueeze_822, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_7, (96, ), (1, ))
    assert_size_stride(primals_9, (96, ), (1, ))
    assert_size_stride(primals_11, (24, ), (1, ))
    assert_size_stride(primals_13, (144, ), (1, ))
    assert_size_stride(primals_15, (144, ), (1, ))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_19, (144, ), (1, ))
    assert_size_stride(primals_21, (144, ), (1, ))
    assert_size_stride(primals_23, (32, ), (1, ))
    assert_size_stride(primals_25, (192, ), (1, ))
    assert_size_stride(primals_27, (192, ), (1, ))
    assert_size_stride(primals_29, (32, ), (1, ))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_33, (192, ), (1, ))
    assert_size_stride(primals_35, (32, ), (1, ))
    assert_size_stride(primals_37, (192, ), (1, ))
    assert_size_stride(primals_39, (192, ), (1, ))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_43, (384, ), (1, ))
    assert_size_stride(primals_45, (384, ), (1, ))
    assert_size_stride(primals_47, (64, ), (1, ))
    assert_size_stride(primals_49, (384, ), (1, ))
    assert_size_stride(primals_51, (384, ), (1, ))
    assert_size_stride(primals_53, (64, ), (1, ))
    assert_size_stride(primals_55, (384, ), (1, ))
    assert_size_stride(primals_57, (384, ), (1, ))
    assert_size_stride(primals_59, (64, ), (1, ))
    assert_size_stride(primals_61, (384, ), (1, ))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_65, (96, ), (1, ))
    assert_size_stride(primals_67, (576, ), (1, ))
    assert_size_stride(primals_69, (576, ), (1, ))
    assert_size_stride(primals_71, (96, ), (1, ))
    assert_size_stride(primals_73, (576, ), (1, ))
    assert_size_stride(primals_75, (576, ), (1, ))
    assert_size_stride(primals_77, (96, ), (1, ))
    assert_size_stride(primals_79, (576, ), (1, ))
    assert_size_stride(primals_81, (576, ), (1, ))
    assert_size_stride(primals_83, (160, ), (1, ))
    assert_size_stride(primals_85, (960, ), (1, ))
    assert_size_stride(primals_87, (960, ), (1, ))
    assert_size_stride(primals_89, (160, ), (1, ))
    assert_size_stride(primals_91, (960, ), (1, ))
    assert_size_stride(primals_93, (960, ), (1, ))
    assert_size_stride(primals_95, (160, ), (1, ))
    assert_size_stride(primals_97, (960, ), (1, ))
    assert_size_stride(primals_99, (960, ), (1, ))
    assert_size_stride(primals_101, (320, ), (1, ))
    assert_size_stride(primals_103, (1280, ), (1, ))
    assert_size_stride(primals_105, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_106, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_107, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_108, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_109, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_110, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_111, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_112, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_113, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_114, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_115, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_116, (32, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_117, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_118, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_119, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_120, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_121, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_122, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_123, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_124, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_125, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_126, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_127, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_128, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_129, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_130, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_131, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_132, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_133, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_134, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_135, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_136, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_137, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_138, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_139, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_140, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_141, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_142, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_143, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_144, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_145, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_146, (160, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_147, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_148, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_149, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_150, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_151, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_152, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_153, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_154, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_155, (320, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_156, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_315, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(clamp_max, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_1, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(clamp_max_1, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_7, (16, ), (1, ))
    assert_size_stride(add_14, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_3, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(squeeze_10, (96, ), (1, ))
    assert_size_stride(clamp_max_2, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(convolution_4, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(squeeze_13, (96, ), (1, ))
    assert_size_stride(clamp_max_3, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(convolution_5, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_16, (24, ), (1, ))
    assert_size_stride(add_29, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_6, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(squeeze_19, (144, ), (1, ))
    assert_size_stride(clamp_max_4, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(convolution_7, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(squeeze_22, (144, ), (1, ))
    assert_size_stride(clamp_max_5, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(convolution_8, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_25, (24, ), (1, ))
    assert_size_stride(add_45, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_9, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(squeeze_28, (144, ), (1, ))
    assert_size_stride(clamp_max_6, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(convolution_10, (8, 144, 28, 28), (112896, 1, 4032, 144))
    assert_size_stride(squeeze_31, (144, ), (1, ))
    assert_size_stride(clamp_max_7, (8, 144, 28, 28), (112896, 1, 4032, 144))
    assert_size_stride(convolution_11, (8, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(squeeze_34, (32, ), (1, ))
    assert_size_stride(add_60, (8, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(convolution_12, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_37, (192, ), (1, ))
    assert_size_stride(clamp_max_8, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_13, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_40, (192, ), (1, ))
    assert_size_stride(clamp_max_9, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_14, (8, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(squeeze_43, (32, ), (1, ))
    assert_size_stride(add_76, (8, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(convolution_15, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_46, (192, ), (1, ))
    assert_size_stride(clamp_max_10, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_16, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_49, (192, ), (1, ))
    assert_size_stride(clamp_max_11, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_17, (8, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(squeeze_52, (32, ), (1, ))
    assert_size_stride(add_92, (8, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(convolution_18, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_55, (192, ), (1, ))
    assert_size_stride(clamp_max_12, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_19, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(squeeze_58, (192, ), (1, ))
    assert_size_stride(clamp_max_13, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_20, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(squeeze_61, (64, ), (1, ))
    assert_size_stride(add_107, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(convolution_21, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_64, (384, ), (1, ))
    assert_size_stride(clamp_max_14, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_22, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_67, (384, ), (1, ))
    assert_size_stride(clamp_max_15, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_23, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(squeeze_70, (64, ), (1, ))
    assert_size_stride(add_123, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(convolution_24, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_73, (384, ), (1, ))
    assert_size_stride(clamp_max_16, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_25, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_76, (384, ), (1, ))
    assert_size_stride(clamp_max_17, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_26, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(squeeze_79, (64, ), (1, ))
    assert_size_stride(add_139, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(convolution_27, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_82, (384, ), (1, ))
    assert_size_stride(clamp_max_18, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_28, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_85, (384, ), (1, ))
    assert_size_stride(clamp_max_19, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_29, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(squeeze_88, (64, ), (1, ))
    assert_size_stride(add_155, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(convolution_30, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_91, (384, ), (1, ))
    assert_size_stride(clamp_max_20, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_31, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_94, (384, ), (1, ))
    assert_size_stride(clamp_max_21, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_32, (8, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(squeeze_97, (96, ), (1, ))
    assert_size_stride(add_170, (8, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(convolution_33, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(squeeze_100, (576, ), (1, ))
    assert_size_stride(clamp_max_22, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_34, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(squeeze_103, (576, ), (1, ))
    assert_size_stride(clamp_max_23, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_35, (8, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(squeeze_106, (96, ), (1, ))
    assert_size_stride(add_186, (8, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(convolution_36, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(squeeze_109, (576, ), (1, ))
    assert_size_stride(clamp_max_24, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_37, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(squeeze_112, (576, ), (1, ))
    assert_size_stride(clamp_max_25, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_38, (8, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(squeeze_115, (96, ), (1, ))
    assert_size_stride(add_202, (8, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(convolution_39, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(squeeze_118, (576, ), (1, ))
    assert_size_stride(clamp_max_26, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_40, (8, 576, 7, 7), (28224, 1, 4032, 576))
    assert_size_stride(squeeze_121, (576, ), (1, ))
    assert_size_stride(clamp_max_27, (8, 576, 7, 7), (28224, 1, 4032, 576))
    assert_size_stride(convolution_41, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(squeeze_124, (160, ), (1, ))
    assert_size_stride(add_217, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_42, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_127, (960, ), (1, ))
    assert_size_stride(clamp_max_28, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_43, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_130, (960, ), (1, ))
    assert_size_stride(clamp_max_29, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_44, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(squeeze_133, (160, ), (1, ))
    assert_size_stride(add_233, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_45, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_136, (960, ), (1, ))
    assert_size_stride(clamp_max_30, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_46, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_139, (960, ), (1, ))
    assert_size_stride(clamp_max_31, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_47, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(squeeze_142, (160, ), (1, ))
    assert_size_stride(add_249, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_48, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_145, (960, ), (1, ))
    assert_size_stride(clamp_max_32, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_49, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_148, (960, ), (1, ))
    assert_size_stride(clamp_max_33, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_50, (8, 320, 7, 7), (15680, 1, 2240, 320))
    assert_size_stride(squeeze_151, (320, ), (1, ))
    assert_size_stride(add_264, (8, 320, 7, 7), (15680, 1, 2240, 320))
    assert_size_stride(convolution_51, (8, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(squeeze_154, (1280, ), (1, ))
    assert_size_stride(view, (8, 1280), (1280, 1))
    assert_size_stride(permute_1, (1000, 1280), (1280, 1))
    assert_size_stride(bitwise_or, (8, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(unsqueeze_210, (1, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(unsqueeze_222, (1, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(bitwise_or_1, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(unsqueeze_234, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(bitwise_or_2, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(unsqueeze_246, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(bitwise_or_3, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(unsqueeze_270, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(bitwise_or_4, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(unsqueeze_282, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_294, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(bitwise_or_5, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(unsqueeze_306, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(bitwise_or_6, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(unsqueeze_318, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(bitwise_or_7, (8, 576, 7, 7), (28224, 1, 4032, 576))
    assert_size_stride(unsqueeze_342, (1, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(bitwise_or_8, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(unsqueeze_354, (1, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(unsqueeze_366, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(bitwise_or_9, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(unsqueeze_378, (1, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(bitwise_or_10, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(unsqueeze_390, (1, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(bitwise_or_11, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(unsqueeze_414, (1, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(bitwise_or_12, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(unsqueeze_426, (1, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(unsqueeze_438, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(bitwise_or_13, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(unsqueeze_450, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(bitwise_or_14, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(unsqueeze_462, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(bitwise_or_15, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(unsqueeze_486, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(bitwise_or_16, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(unsqueeze_498, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_510, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(bitwise_or_17, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(unsqueeze_522, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(bitwise_or_18, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(unsqueeze_534, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_546, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(bitwise_or_19, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(unsqueeze_558, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(bitwise_or_20, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(unsqueeze_570, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_582, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(bitwise_or_21, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(unsqueeze_594, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(bitwise_or_22, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(unsqueeze_606, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_618, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(bitwise_or_23, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(unsqueeze_630, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(bitwise_or_24, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(unsqueeze_642, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_654, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(bitwise_or_25, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(unsqueeze_666, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(bitwise_or_26, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(unsqueeze_678, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_690, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(bitwise_or_27, (8, 144, 28, 28), (112896, 1, 4032, 144))
    assert_size_stride(unsqueeze_702, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(bitwise_or_28, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(unsqueeze_714, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(unsqueeze_726, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(bitwise_or_29, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(unsqueeze_738, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(bitwise_or_30, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(unsqueeze_750, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(unsqueeze_762, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(bitwise_or_31, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(unsqueeze_774, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(bitwise_or_32, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(unsqueeze_786, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_798, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(bitwise_or_33, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(unsqueeze_810, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(bitwise_or_34, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(unsqueeze_822, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), view, out=buf1)
        del view
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty_strided((1280, 4), (1, 1280), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1280, 4), (1, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_div_hardtanh_backward_native_batch_norm_backward_1.run(bitwise_or, buf0, convolution_51, unsqueeze_210, buf3, buf5, 5120, 98, grid=grid(5120), stream=stream0)
        buf4 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardtanh_backward_native_batch_norm_backward_2.run(buf3, buf4, 1280, 4, grid=grid(1280), stream=stream0)
        del buf3
        buf6 = empty((1280, ), device='cuda', dtype=torch.float32)
        buf7 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardtanh_backward_native_batch_norm_backward_3.run(buf5, squeeze_154, buf6, buf7, 1280, 4, grid=grid(1280), stream=stream0)
        del buf5
        buf8 = empty_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_div_hardtanh_backward_native_batch_norm_backward_4.run(bitwise_or, buf0, convolution_51, unsqueeze_210, buf6, squeeze_154, buf4, primals_103, buf8, 501760, grid=grid(501760), stream=stream0)
        del bitwise_or
        del buf0
        del convolution_51
        del primals_103
        del squeeze_154
        del unsqueeze_210
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf9 = aten.convolution_backward(buf8, add_264, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_264
        del buf8
        del primals_156
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        buf12 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_5.run(buf10, buf12, 320, 392, grid=grid(320), stream=stream0)
        buf13 = reinterpret_tensor(buf6, (320, 4), (1, 320), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_6.run(buf10, convolution_50, unsqueeze_222, buf13, 1280, 98, grid=grid(1280), stream=stream0)
        buf14 = empty((320, ), device='cuda', dtype=torch.float32)
        buf15 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_7.run(buf13, squeeze_151, buf14, buf15, 320, 4, grid=grid(320), stream=stream0)
        del buf13
        buf16 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_8.run(buf16, convolution_50, unsqueeze_222, buf14, squeeze_151, buf12, primals_101, 2560, 49, grid=grid(2560, 49), stream=stream0)
        del buf14
        del convolution_50
        del primals_101
        del squeeze_151
        del unsqueeze_222
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf17 = aten.convolution_backward(buf16, clamp_max_33, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf16
        del clamp_max_33
        del primals_155
        buf18 = buf17[0]
        buf19 = buf17[1]
        del buf17
        buf20 = empty_strided((960, 4), (1, 960), device='cuda', dtype=torch.float32)
        buf22 = empty_strided((960, 4), (1, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_9.run(bitwise_or_1, buf18, convolution_49, unsqueeze_234, buf20, buf22, 3840, 98, grid=grid(3840), stream=stream0)
        buf21 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_10.run(buf20, buf21, 960, 4, grid=grid(960), stream=stream0)
        buf23 = empty((960, ), device='cuda', dtype=torch.float32)
        buf24 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_11.run(buf22, squeeze_148, buf23, buf24, 960, 4, grid=grid(960), stream=stream0)
        buf25 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_12.run(bitwise_or_1, buf18, convolution_49, unsqueeze_234, buf23, squeeze_148, buf21, primals_99, buf25, 392, 960, grid=grid(392, 960), stream=stream0)
        del bitwise_or_1
        del buf18
        del convolution_49
        del primals_99
        del squeeze_148
        del unsqueeze_234
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf26 = aten.convolution_backward(buf25, clamp_max_32, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 960, [True, True, False])
        del clamp_max_32
        del primals_154
        buf27 = buf26[0]
        buf28 = buf26[1]
        del buf26
        buf29 = buf22; del buf22  # reuse
        buf31 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_9.run(bitwise_or_2, buf27, convolution_48, unsqueeze_246, buf29, buf31, 3840, 98, grid=grid(3840), stream=stream0)
        buf30 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_10.run(buf29, buf30, 960, 4, grid=grid(960), stream=stream0)
        buf32 = empty((960, ), device='cuda', dtype=torch.float32)
        buf33 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_11.run(buf31, squeeze_145, buf32, buf33, 960, 4, grid=grid(960), stream=stream0)
        buf34 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_12.run(bitwise_or_2, buf27, convolution_48, unsqueeze_246, buf32, squeeze_145, buf30, primals_97, buf34, 392, 960, grid=grid(392, 960), stream=stream0)
        del bitwise_or_2
        del buf27
        del convolution_48
        del primals_97
        del squeeze_145
        del unsqueeze_246
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf35 = aten.convolution_backward(buf34, add_249, primals_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_249
        del primals_153
        buf36 = buf35[0]
        buf37 = buf35[1]
        del buf35
        buf38 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_13.run(buf36, buf38, 160, 392, grid=grid(160), stream=stream0)
        buf39 = empty_strided((160, 4), (1, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_14.run(buf36, convolution_47, unsqueeze_258, buf39, 640, 98, grid=grid(640), stream=stream0)
        buf40 = empty((160, ), device='cuda', dtype=torch.float32)
        buf41 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_15.run(buf39, squeeze_142, buf40, buf41, 160, 4, grid=grid(160), stream=stream0)
        del buf39
        buf42 = empty((8, 160, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_16.run(buf36, convolution_47, unsqueeze_258, buf40, squeeze_142, buf38, primals_95, buf42, 1280, 49, grid=grid(1280, 49), stream=stream0)
        del convolution_47
        del primals_95
        del squeeze_142
        del unsqueeze_258
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf43 = aten.convolution_backward(buf42, clamp_max_31, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clamp_max_31
        del primals_152
        buf44 = buf43[0]
        buf45 = buf43[1]
        del buf43
        buf46 = buf31; del buf31  # reuse
        buf48 = buf29; del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_9.run(bitwise_or_3, buf44, convolution_46, unsqueeze_270, buf46, buf48, 3840, 98, grid=grid(3840), stream=stream0)
        buf47 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_10.run(buf46, buf47, 960, 4, grid=grid(960), stream=stream0)
        buf49 = empty((960, ), device='cuda', dtype=torch.float32)
        buf50 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_11.run(buf48, squeeze_139, buf49, buf50, 960, 4, grid=grid(960), stream=stream0)
        buf51 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_12.run(bitwise_or_3, buf44, convolution_46, unsqueeze_270, buf49, squeeze_139, buf47, primals_93, buf51, 392, 960, grid=grid(392, 960), stream=stream0)
        del bitwise_or_3
        del buf44
        del convolution_46
        del primals_93
        del squeeze_139
        del unsqueeze_270
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf52 = aten.convolution_backward(buf51, clamp_max_30, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 960, [True, True, False])
        del clamp_max_30
        del primals_151
        buf53 = buf52[0]
        buf54 = buf52[1]
        del buf52
        buf55 = buf48; del buf48  # reuse
        buf57 = buf46; del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_9.run(bitwise_or_4, buf53, convolution_45, unsqueeze_282, buf55, buf57, 3840, 98, grid=grid(3840), stream=stream0)
        buf56 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_10.run(buf55, buf56, 960, 4, grid=grid(960), stream=stream0)
        buf58 = empty((960, ), device='cuda', dtype=torch.float32)
        buf59 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_11.run(buf57, squeeze_136, buf58, buf59, 960, 4, grid=grid(960), stream=stream0)
        buf60 = buf51; del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_12.run(bitwise_or_4, buf53, convolution_45, unsqueeze_282, buf58, squeeze_136, buf56, primals_91, buf60, 392, 960, grid=grid(392, 960), stream=stream0)
        del bitwise_or_4
        del buf53
        del convolution_45
        del primals_91
        del squeeze_136
        del unsqueeze_282
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf61 = aten.convolution_backward(buf60, add_233, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_233
        del primals_150
        buf62 = buf61[0]
        buf63 = buf61[1]
        del buf61
        buf64 = buf40; del buf40  # reuse
        buf65 = empty((160, ), device='cuda', dtype=torch.float32)
        buf66 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_17.run(buf36, buf62, convolution_44, unsqueeze_294, squeeze_133, buf64, buf65, buf66, 160, 392, grid=grid(160), stream=stream0)
        buf67 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_18.run(buf36, buf62, convolution_44, unsqueeze_294, buf65, squeeze_133, buf64, primals_89, buf67, 1280, 49, grid=grid(1280, 49), stream=stream0)
        del convolution_44
        del primals_89
        del squeeze_133
        del unsqueeze_294
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf68 = aten.convolution_backward(buf67, clamp_max_29, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf67
        del clamp_max_29
        del primals_149
        buf69 = buf68[0]
        buf70 = buf68[1]
        del buf68
        buf71 = buf57; del buf57  # reuse
        buf73 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_9.run(bitwise_or_5, buf69, convolution_43, unsqueeze_306, buf71, buf73, 3840, 98, grid=grid(3840), stream=stream0)
        buf72 = buf58; del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_10.run(buf71, buf72, 960, 4, grid=grid(960), stream=stream0)
        buf74 = empty((960, ), device='cuda', dtype=torch.float32)
        buf75 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_11.run(buf73, squeeze_130, buf74, buf75, 960, 4, grid=grid(960), stream=stream0)
        buf76 = buf60; del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_12.run(bitwise_or_5, buf69, convolution_43, unsqueeze_306, buf74, squeeze_130, buf72, primals_87, buf76, 392, 960, grid=grid(392, 960), stream=stream0)
        del bitwise_or_5
        del buf69
        del convolution_43
        del primals_87
        del squeeze_130
        del unsqueeze_306
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf77 = aten.convolution_backward(buf76, clamp_max_28, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 960, [True, True, False])
        del clamp_max_28
        del primals_148
        buf78 = buf77[0]
        buf79 = buf77[1]
        del buf77
        buf80 = buf73; del buf73  # reuse
        buf82 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_9.run(bitwise_or_6, buf78, convolution_42, unsqueeze_318, buf80, buf82, 3840, 98, grid=grid(3840), stream=stream0)
        buf81 = buf74; del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_10.run(buf80, buf81, 960, 4, grid=grid(960), stream=stream0)
        del buf80
        buf83 = empty((960, ), device='cuda', dtype=torch.float32)
        buf84 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_11.run(buf82, squeeze_127, buf83, buf84, 960, 4, grid=grid(960), stream=stream0)
        del buf82
        buf85 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_12.run(bitwise_or_6, buf78, convolution_42, unsqueeze_318, buf83, squeeze_127, buf81, primals_85, buf85, 392, 960, grid=grid(392, 960), stream=stream0)
        del bitwise_or_6
        del buf78
        del buf83
        del convolution_42
        del primals_85
        del squeeze_127
        del unsqueeze_318
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf86 = aten.convolution_backward(buf85, add_217, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_217
        del buf85
        del primals_147
        buf87 = buf86[0]
        buf88 = buf86[1]
        del buf86
        buf89 = buf65; del buf65  # reuse
        buf90 = empty((160, ), device='cuda', dtype=torch.float32)
        buf92 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_19.run(buf36, buf62, buf87, convolution_41, unsqueeze_330, squeeze_124, buf89, buf90, buf92, 160, 392, grid=grid(160), stream=stream0)
        buf91 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_20.run(buf91, buf62, buf87, convolution_41, unsqueeze_330, buf90, squeeze_124, buf89, primals_83, 1280, 49, grid=grid(1280, 49), stream=stream0)
        del buf62
        del buf87
        del buf90
        del convolution_41
        del primals_83
        del squeeze_124
        del unsqueeze_330
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf93 = aten.convolution_backward(buf91, clamp_max_27, primals_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf91
        del clamp_max_27
        del primals_146
        buf94 = buf93[0]
        buf95 = buf93[1]
        del buf93
        buf96 = empty_strided((576, 4), (1, 576), device='cuda', dtype=torch.float32)
        buf98 = empty_strided((576, 4), (1, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_21.run(bitwise_or_7, buf94, convolution_40, unsqueeze_342, buf96, buf98, 2304, 98, grid=grid(2304), stream=stream0)
        buf97 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_22.run(buf96, buf97, 576, 4, grid=grid(576), stream=stream0)
        del buf96
        buf99 = empty((576, ), device='cuda', dtype=torch.float32)
        buf100 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_23.run(buf98, squeeze_121, buf99, buf100, 576, 4, grid=grid(576), stream=stream0)
        del buf98
        buf101 = empty_strided((8, 576, 7, 7), (28224, 1, 4032, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_24.run(bitwise_or_7, buf94, convolution_40, unsqueeze_342, buf99, squeeze_121, buf97, primals_81, buf101, 392, 576, grid=grid(392, 576), stream=stream0)
        del bitwise_or_7
        del buf94
        del convolution_40
        del primals_81
        del squeeze_121
        del unsqueeze_342
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf102 = aten.convolution_backward(buf101, clamp_max_26, primals_145, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False])
        del buf101
        del clamp_max_26
        del primals_145
        buf103 = buf102[0]
        buf104 = buf102[1]
        del buf102
        buf105 = empty((576, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_25.run(bitwise_or_8, buf103, buf105, 7488, 121, grid=grid(7488), stream=stream0)
        buf106 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_26.run(buf105, buf106, 576, 13, grid=grid(576), stream=stream0)
        buf107 = reinterpret_tensor(buf105, (576, 13), (1, 576), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_27.run(bitwise_or_8, buf103, convolution_39, unsqueeze_354, buf107, 7488, 121, grid=grid(7488), stream=stream0)
        buf108 = empty((576, ), device='cuda', dtype=torch.float32)
        buf109 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_28.run(buf107, squeeze_118, buf108, buf109, 576, 13, grid=grid(576), stream=stream0)
        buf110 = empty_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_29.run(bitwise_or_8, buf103, convolution_39, unsqueeze_354, buf108, squeeze_118, buf106, primals_79, buf110, 1568, 576, grid=grid(1568, 576), stream=stream0)
        del bitwise_or_8
        del buf103
        del convolution_39
        del primals_79
        del squeeze_118
        del unsqueeze_354
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf111 = aten.convolution_backward(buf110, add_202, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_202
        del primals_144
        buf112 = buf111[0]
        buf113 = buf111[1]
        del buf111
        buf114 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_30.run(buf112, buf114, 96, 1568, grid=grid(96), stream=stream0)
        buf115 = empty((96, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_31.run(buf112, convolution_38, unsqueeze_366, buf115, 1248, 121, grid=grid(1248), stream=stream0)
        buf116 = empty((96, ), device='cuda', dtype=torch.float32)
        buf117 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_32.run(buf115, squeeze_115, buf116, buf117, 96, 13, grid=grid(96), stream=stream0)
        del buf115
        buf118 = empty((8, 96, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_33.run(buf112, convolution_38, unsqueeze_366, buf116, squeeze_115, buf114, primals_77, buf118, 768, 196, grid=grid(768, 196), stream=stream0)
        del convolution_38
        del primals_77
        del squeeze_115
        del unsqueeze_366
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf119 = aten.convolution_backward(buf118, clamp_max_25, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clamp_max_25
        del primals_143
        buf120 = buf119[0]
        buf121 = buf119[1]
        del buf119
        buf122 = reinterpret_tensor(buf107, (576, 13), (13, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_25.run(bitwise_or_9, buf120, buf122, 7488, 121, grid=grid(7488), stream=stream0)
        buf123 = buf108; del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_26.run(buf122, buf123, 576, 13, grid=grid(576), stream=stream0)
        buf124 = reinterpret_tensor(buf122, (576, 13), (1, 576), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_27.run(bitwise_or_9, buf120, convolution_37, unsqueeze_378, buf124, 7488, 121, grid=grid(7488), stream=stream0)
        buf125 = empty((576, ), device='cuda', dtype=torch.float32)
        buf126 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_28.run(buf124, squeeze_112, buf125, buf126, 576, 13, grid=grid(576), stream=stream0)
        buf127 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_29.run(bitwise_or_9, buf120, convolution_37, unsqueeze_378, buf125, squeeze_112, buf123, primals_75, buf127, 1568, 576, grid=grid(1568, 576), stream=stream0)
        del bitwise_or_9
        del buf120
        del convolution_37
        del primals_75
        del squeeze_112
        del unsqueeze_378
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf128 = aten.convolution_backward(buf127, clamp_max_24, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False])
        del clamp_max_24
        del primals_142
        buf129 = buf128[0]
        buf130 = buf128[1]
        del buf128
        buf131 = reinterpret_tensor(buf124, (576, 13), (13, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_25.run(bitwise_or_10, buf129, buf131, 7488, 121, grid=grid(7488), stream=stream0)
        buf132 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_26.run(buf131, buf132, 576, 13, grid=grid(576), stream=stream0)
        buf133 = reinterpret_tensor(buf131, (576, 13), (1, 576), 0); del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_27.run(bitwise_or_10, buf129, convolution_36, unsqueeze_390, buf133, 7488, 121, grid=grid(7488), stream=stream0)
        buf134 = empty((576, ), device='cuda', dtype=torch.float32)
        buf135 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_28.run(buf133, squeeze_109, buf134, buf135, 576, 13, grid=grid(576), stream=stream0)
        buf136 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_29.run(bitwise_or_10, buf129, convolution_36, unsqueeze_390, buf134, squeeze_109, buf132, primals_73, buf136, 1568, 576, grid=grid(1568, 576), stream=stream0)
        del bitwise_or_10
        del buf129
        del convolution_36
        del primals_73
        del squeeze_109
        del unsqueeze_390
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf137 = aten.convolution_backward(buf136, add_186, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_186
        del primals_141
        buf138 = buf137[0]
        buf139 = buf137[1]
        del buf137
        buf140 = buf116; del buf116  # reuse
        buf141 = empty((96, ), device='cuda', dtype=torch.float32)
        buf142 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_34.run(buf112, buf138, convolution_35, unsqueeze_402, squeeze_106, buf140, buf141, buf142, 96, 1568, grid=grid(96), stream=stream0)
        buf143 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_35.run(buf112, buf138, convolution_35, unsqueeze_402, buf141, squeeze_106, buf140, primals_71, buf143, 768, 196, grid=grid(768, 196), stream=stream0)
        del convolution_35
        del primals_71
        del squeeze_106
        del unsqueeze_402
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf144 = aten.convolution_backward(buf143, clamp_max_23, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf143
        del clamp_max_23
        del primals_140
        buf145 = buf144[0]
        buf146 = buf144[1]
        del buf144
        buf147 = reinterpret_tensor(buf133, (576, 13), (13, 1), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_25.run(bitwise_or_11, buf145, buf147, 7488, 121, grid=grid(7488), stream=stream0)
        buf148 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_26.run(buf147, buf148, 576, 13, grid=grid(576), stream=stream0)
        buf149 = reinterpret_tensor(buf147, (576, 13), (1, 576), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_27.run(bitwise_or_11, buf145, convolution_34, unsqueeze_414, buf149, 7488, 121, grid=grid(7488), stream=stream0)
        buf150 = empty((576, ), device='cuda', dtype=torch.float32)
        buf151 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_28.run(buf149, squeeze_103, buf150, buf151, 576, 13, grid=grid(576), stream=stream0)
        buf152 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_29.run(bitwise_or_11, buf145, convolution_34, unsqueeze_414, buf150, squeeze_103, buf148, primals_69, buf152, 1568, 576, grid=grid(1568, 576), stream=stream0)
        del bitwise_or_11
        del buf145
        del convolution_34
        del primals_69
        del squeeze_103
        del unsqueeze_414
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf153 = aten.convolution_backward(buf152, clamp_max_22, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False])
        del clamp_max_22
        del primals_139
        buf154 = buf153[0]
        buf155 = buf153[1]
        del buf153
        buf156 = reinterpret_tensor(buf149, (576, 13), (13, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_25.run(bitwise_or_12, buf154, buf156, 7488, 121, grid=grid(7488), stream=stream0)
        buf157 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_26.run(buf156, buf157, 576, 13, grid=grid(576), stream=stream0)
        buf158 = reinterpret_tensor(buf156, (576, 13), (1, 576), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_27.run(bitwise_or_12, buf154, convolution_33, unsqueeze_426, buf158, 7488, 121, grid=grid(7488), stream=stream0)
        buf159 = empty((576, ), device='cuda', dtype=torch.float32)
        buf160 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_28.run(buf158, squeeze_100, buf159, buf160, 576, 13, grid=grid(576), stream=stream0)
        del buf158
        buf161 = buf152; del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_29.run(bitwise_or_12, buf154, convolution_33, unsqueeze_426, buf159, squeeze_100, buf157, primals_67, buf161, 1568, 576, grid=grid(1568, 576), stream=stream0)
        del bitwise_or_12
        del buf154
        del buf159
        del convolution_33
        del primals_67
        del squeeze_100
        del unsqueeze_426
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf162 = aten.convolution_backward(buf161, add_170, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_170
        del primals_138
        buf163 = buf162[0]
        buf164 = buf162[1]
        del buf162
        buf165 = buf141; del buf141  # reuse
        buf166 = empty((96, ), device='cuda', dtype=torch.float32)
        buf168 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_36.run(buf112, buf138, buf163, convolution_32, unsqueeze_438, squeeze_97, buf165, buf166, buf168, 96, 1568, grid=grid(96), stream=stream0)
        buf167 = buf112; del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_37.run(buf167, buf138, buf163, convolution_32, unsqueeze_438, buf166, squeeze_97, buf165, primals_65, 768, 196, grid=grid(768, 196), stream=stream0)
        del buf138
        del buf163
        del convolution_32
        del primals_65
        del squeeze_97
        del unsqueeze_438
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf169 = aten.convolution_backward(buf167, clamp_max_21, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf167
        del clamp_max_21
        del primals_137
        buf170 = buf169[0]
        buf171 = buf169[1]
        del buf169
        buf172 = empty((384, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_38.run(bitwise_or_13, buf170, buf172, 4992, 121, grid=grid(4992), stream=stream0)
        buf173 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_39.run(buf172, buf173, 384, 13, grid=grid(384), stream=stream0)
        buf174 = reinterpret_tensor(buf172, (384, 13), (1, 384), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_40.run(bitwise_or_13, buf170, convolution_31, unsqueeze_450, buf174, 4992, 121, grid=grid(4992), stream=stream0)
        buf175 = empty((384, ), device='cuda', dtype=torch.float32)
        buf176 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_41.run(buf174, squeeze_94, buf175, buf176, 384, 13, grid=grid(384), stream=stream0)
        buf177 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_42.run(bitwise_or_13, buf170, convolution_31, unsqueeze_450, buf175, squeeze_94, buf173, primals_63, buf177, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del bitwise_or_13
        del buf170
        del convolution_31
        del primals_63
        del squeeze_94
        del unsqueeze_450
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf178 = aten.convolution_backward(buf177, clamp_max_20, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False])
        del clamp_max_20
        del primals_136
        buf179 = buf178[0]
        buf180 = buf178[1]
        del buf178
        buf181 = reinterpret_tensor(buf174, (384, 13), (13, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_38.run(bitwise_or_14, buf179, buf181, 4992, 121, grid=grid(4992), stream=stream0)
        buf182 = buf175; del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_39.run(buf181, buf182, 384, 13, grid=grid(384), stream=stream0)
        buf183 = reinterpret_tensor(buf181, (384, 13), (1, 384), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_40.run(bitwise_or_14, buf179, convolution_30, unsqueeze_462, buf183, 4992, 121, grid=grid(4992), stream=stream0)
        buf184 = empty((384, ), device='cuda', dtype=torch.float32)
        buf185 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_41.run(buf183, squeeze_91, buf184, buf185, 384, 13, grid=grid(384), stream=stream0)
        buf186 = buf177; del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_42.run(bitwise_or_14, buf179, convolution_30, unsqueeze_462, buf184, squeeze_91, buf182, primals_61, buf186, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del bitwise_or_14
        del buf179
        del convolution_30
        del primals_61
        del squeeze_91
        del unsqueeze_462
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf187 = aten.convolution_backward(buf186, add_155, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_155
        del primals_135
        buf188 = buf187[0]
        buf189 = buf187[1]
        del buf187
        buf190 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_43.run(buf188, buf190, 64, 1568, grid=grid(64), stream=stream0)
        buf191 = empty((64, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_44.run(buf188, convolution_29, unsqueeze_474, buf191, 832, 121, grid=grid(832), stream=stream0)
        buf192 = empty((64, ), device='cuda', dtype=torch.float32)
        buf193 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_45.run(buf191, squeeze_88, buf192, buf193, 64, 13, grid=grid(64), stream=stream0)
        del buf191
        buf194 = empty((8, 64, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_46.run(buf188, convolution_29, unsqueeze_474, buf192, squeeze_88, buf190, primals_59, buf194, 512, 196, grid=grid(512, 196), stream=stream0)
        del convolution_29
        del primals_59
        del squeeze_88
        del unsqueeze_474
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf195 = aten.convolution_backward(buf194, clamp_max_19, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clamp_max_19
        del primals_134
        buf196 = buf195[0]
        buf197 = buf195[1]
        del buf195
        buf198 = reinterpret_tensor(buf183, (384, 13), (13, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_38.run(bitwise_or_15, buf196, buf198, 4992, 121, grid=grid(4992), stream=stream0)
        buf199 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_39.run(buf198, buf199, 384, 13, grid=grid(384), stream=stream0)
        buf200 = reinterpret_tensor(buf198, (384, 13), (1, 384), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_40.run(bitwise_or_15, buf196, convolution_28, unsqueeze_486, buf200, 4992, 121, grid=grid(4992), stream=stream0)
        buf201 = empty((384, ), device='cuda', dtype=torch.float32)
        buf202 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_41.run(buf200, squeeze_85, buf201, buf202, 384, 13, grid=grid(384), stream=stream0)
        buf203 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_42.run(bitwise_or_15, buf196, convolution_28, unsqueeze_486, buf201, squeeze_85, buf199, primals_57, buf203, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del bitwise_or_15
        del buf196
        del convolution_28
        del primals_57
        del squeeze_85
        del unsqueeze_486
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf204 = aten.convolution_backward(buf203, clamp_max_18, primals_133, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False])
        del clamp_max_18
        del primals_133
        buf205 = buf204[0]
        buf206 = buf204[1]
        del buf204
        buf207 = reinterpret_tensor(buf200, (384, 13), (13, 1), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_38.run(bitwise_or_16, buf205, buf207, 4992, 121, grid=grid(4992), stream=stream0)
        buf208 = buf201; del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_39.run(buf207, buf208, 384, 13, grid=grid(384), stream=stream0)
        buf209 = reinterpret_tensor(buf207, (384, 13), (1, 384), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_40.run(bitwise_or_16, buf205, convolution_27, unsqueeze_498, buf209, 4992, 121, grid=grid(4992), stream=stream0)
        buf210 = empty((384, ), device='cuda', dtype=torch.float32)
        buf211 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_41.run(buf209, squeeze_82, buf210, buf211, 384, 13, grid=grid(384), stream=stream0)
        buf212 = buf203; del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_42.run(bitwise_or_16, buf205, convolution_27, unsqueeze_498, buf210, squeeze_82, buf208, primals_55, buf212, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del bitwise_or_16
        del buf205
        del convolution_27
        del primals_55
        del squeeze_82
        del unsqueeze_498
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf213 = aten.convolution_backward(buf212, add_139, primals_132, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_139
        del primals_132
        buf214 = buf213[0]
        buf215 = buf213[1]
        del buf213
        buf216 = buf192; del buf192  # reuse
        buf217 = empty((64, ), device='cuda', dtype=torch.float32)
        buf218 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_47.run(buf188, buf214, convolution_26, unsqueeze_510, squeeze_79, buf216, buf217, buf218, 64, 1568, grid=grid(64), stream=stream0)
        buf219 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_48.run(buf188, buf214, convolution_26, unsqueeze_510, buf217, squeeze_79, buf216, primals_53, buf219, 512, 196, grid=grid(512, 196), stream=stream0)
        del convolution_26
        del primals_53
        del squeeze_79
        del unsqueeze_510
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf220 = aten.convolution_backward(buf219, clamp_max_17, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clamp_max_17
        del primals_131
        buf221 = buf220[0]
        buf222 = buf220[1]
        del buf220
        buf223 = reinterpret_tensor(buf209, (384, 13), (13, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_38.run(bitwise_or_17, buf221, buf223, 4992, 121, grid=grid(4992), stream=stream0)
        buf224 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_39.run(buf223, buf224, 384, 13, grid=grid(384), stream=stream0)
        buf225 = reinterpret_tensor(buf223, (384, 13), (1, 384), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_40.run(bitwise_or_17, buf221, convolution_25, unsqueeze_522, buf225, 4992, 121, grid=grid(4992), stream=stream0)
        buf226 = empty((384, ), device='cuda', dtype=torch.float32)
        buf227 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_41.run(buf225, squeeze_76, buf226, buf227, 384, 13, grid=grid(384), stream=stream0)
        buf228 = buf212; del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_42.run(bitwise_or_17, buf221, convolution_25, unsqueeze_522, buf226, squeeze_76, buf224, primals_51, buf228, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del bitwise_or_17
        del buf221
        del convolution_25
        del primals_51
        del squeeze_76
        del unsqueeze_522
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf229 = aten.convolution_backward(buf228, clamp_max_16, primals_130, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False])
        del clamp_max_16
        del primals_130
        buf230 = buf229[0]
        buf231 = buf229[1]
        del buf229
        buf232 = reinterpret_tensor(buf225, (384, 13), (13, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_38.run(bitwise_or_18, buf230, buf232, 4992, 121, grid=grid(4992), stream=stream0)
        buf233 = buf226; del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_39.run(buf232, buf233, 384, 13, grid=grid(384), stream=stream0)
        buf234 = reinterpret_tensor(buf232, (384, 13), (1, 384), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_40.run(bitwise_or_18, buf230, convolution_24, unsqueeze_534, buf234, 4992, 121, grid=grid(4992), stream=stream0)
        buf235 = empty((384, ), device='cuda', dtype=torch.float32)
        buf236 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_41.run(buf234, squeeze_73, buf235, buf236, 384, 13, grid=grid(384), stream=stream0)
        buf237 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_42.run(bitwise_or_18, buf230, convolution_24, unsqueeze_534, buf235, squeeze_73, buf233, primals_49, buf237, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del bitwise_or_18
        del buf230
        del convolution_24
        del primals_49
        del squeeze_73
        del unsqueeze_534
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf238 = aten.convolution_backward(buf237, add_123, primals_129, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_123
        del primals_129
        buf239 = buf238[0]
        buf240 = buf238[1]
        del buf238
        buf241 = buf217; del buf217  # reuse
        buf242 = empty((64, ), device='cuda', dtype=torch.float32)
        buf244 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_49.run(buf188, buf214, buf239, convolution_23, unsqueeze_546, squeeze_70, buf241, buf242, buf244, 64, 1568, grid=grid(64), stream=stream0)
        buf243 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_50.run(buf188, buf214, buf239, convolution_23, unsqueeze_546, buf242, squeeze_70, buf241, primals_47, buf243, 512, 196, grid=grid(512, 196), stream=stream0)
        del convolution_23
        del primals_47
        del squeeze_70
        del unsqueeze_546
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf245 = aten.convolution_backward(buf243, clamp_max_15, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf243
        del clamp_max_15
        del primals_128
        buf246 = buf245[0]
        buf247 = buf245[1]
        del buf245
        buf248 = reinterpret_tensor(buf234, (384, 13), (13, 1), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_38.run(bitwise_or_19, buf246, buf248, 4992, 121, grid=grid(4992), stream=stream0)
        buf249 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_39.run(buf248, buf249, 384, 13, grid=grid(384), stream=stream0)
        buf250 = reinterpret_tensor(buf248, (384, 13), (1, 384), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_40.run(bitwise_or_19, buf246, convolution_22, unsqueeze_558, buf250, 4992, 121, grid=grid(4992), stream=stream0)
        buf251 = empty((384, ), device='cuda', dtype=torch.float32)
        buf252 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_41.run(buf250, squeeze_67, buf251, buf252, 384, 13, grid=grid(384), stream=stream0)
        buf253 = buf237; del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_42.run(bitwise_or_19, buf246, convolution_22, unsqueeze_558, buf251, squeeze_67, buf249, primals_45, buf253, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del bitwise_or_19
        del buf246
        del convolution_22
        del primals_45
        del squeeze_67
        del unsqueeze_558
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf254 = aten.convolution_backward(buf253, clamp_max_14, primals_127, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False])
        del clamp_max_14
        del primals_127
        buf255 = buf254[0]
        buf256 = buf254[1]
        del buf254
        buf257 = reinterpret_tensor(buf250, (384, 13), (13, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_38.run(bitwise_or_20, buf255, buf257, 4992, 121, grid=grid(4992), stream=stream0)
        buf258 = buf251; del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_39.run(buf257, buf258, 384, 13, grid=grid(384), stream=stream0)
        buf259 = reinterpret_tensor(buf257, (384, 13), (1, 384), 0); del buf257  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_40.run(bitwise_or_20, buf255, convolution_21, unsqueeze_570, buf259, 4992, 121, grid=grid(4992), stream=stream0)
        buf260 = empty((384, ), device='cuda', dtype=torch.float32)
        buf261 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_41.run(buf259, squeeze_64, buf260, buf261, 384, 13, grid=grid(384), stream=stream0)
        del buf259
        buf262 = buf253; del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_42.run(bitwise_or_20, buf255, convolution_21, unsqueeze_570, buf260, squeeze_64, buf258, primals_43, buf262, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del bitwise_or_20
        del buf255
        del buf260
        del convolution_21
        del primals_43
        del squeeze_64
        del unsqueeze_570
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf263 = aten.convolution_backward(buf262, add_107, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_107
        del primals_126
        buf264 = buf263[0]
        buf265 = buf263[1]
        del buf263
        buf266 = buf242; del buf242  # reuse
        buf267 = empty((64, ), device='cuda', dtype=torch.float32)
        buf269 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_51.run(buf188, buf214, buf239, buf264, convolution_20, unsqueeze_582, squeeze_61, buf266, buf267, buf269, 64, 1568, grid=grid(64), stream=stream0)
        buf268 = buf188; del buf188  # reuse
        buf270 = buf268; del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_52.run(buf270, buf214, buf239, buf264, convolution_20, unsqueeze_582, buf267, squeeze_61, buf266, primals_41, 512, 196, grid=grid(512, 196), stream=stream0)
        del buf214
        del buf239
        del buf264
        del buf267
        del convolution_20
        del primals_41
        del squeeze_61
        del unsqueeze_582
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf271 = aten.convolution_backward(buf270, clamp_max_13, primals_125, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf270
        del clamp_max_13
        del primals_125
        buf272 = buf271[0]
        buf273 = buf271[1]
        del buf271
        buf274 = empty((192, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_53.run(bitwise_or_21, buf272, buf274, 2496, 121, grid=grid(2496), stream=stream0)
        buf275 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_54.run(buf274, buf275, 192, 13, grid=grid(192), stream=stream0)
        buf276 = reinterpret_tensor(buf274, (192, 13), (1, 192), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_55.run(bitwise_or_21, buf272, convolution_19, unsqueeze_594, buf276, 2496, 121, grid=grid(2496), stream=stream0)
        buf277 = empty((192, ), device='cuda', dtype=torch.float32)
        buf278 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_56.run(buf276, squeeze_58, buf277, buf278, 192, 13, grid=grid(192), stream=stream0)
        del buf276
        buf279 = empty_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_57.run(bitwise_or_21, buf272, convolution_19, unsqueeze_594, buf277, squeeze_58, buf275, primals_39, buf279, 1568, 192, grid=grid(1568, 192), stream=stream0)
        del bitwise_or_21
        del buf272
        del convolution_19
        del primals_39
        del squeeze_58
        del unsqueeze_594
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf280 = aten.convolution_backward(buf279, clamp_max_12, primals_124, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False])
        del buf279
        del clamp_max_12
        del primals_124
        buf281 = buf280[0]
        buf282 = buf280[1]
        del buf280
        buf283 = empty((192, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_58.run(bitwise_or_22, buf281, buf283, 9408, 128, grid=grid(9408), stream=stream0)
        buf284 = buf277; del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_59.run(buf283, buf284, 192, 49, grid=grid(192), stream=stream0)
        buf285 = reinterpret_tensor(buf283, (192, 49), (1, 192), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_60.run(bitwise_or_22, buf281, convolution_18, unsqueeze_606, buf285, 9408, 128, grid=grid(9408), stream=stream0)
        buf286 = empty((192, ), device='cuda', dtype=torch.float32)
        buf287 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_61.run(buf285, squeeze_55, buf286, buf287, 192, 49, grid=grid(192), stream=stream0)
        buf288 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_62.run(bitwise_or_22, buf281, convolution_18, unsqueeze_606, buf286, squeeze_55, buf284, primals_37, buf288, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del bitwise_or_22
        del buf281
        del convolution_18
        del primals_37
        del squeeze_55
        del unsqueeze_606
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf289 = aten.convolution_backward(buf288, add_92, primals_123, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_92
        del primals_123
        buf290 = buf289[0]
        buf291 = buf289[1]
        del buf289
        buf292 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_63.run(buf290, buf292, 32, 6272, grid=grid(32), stream=stream0)
        buf293 = empty((32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_64.run(buf290, convolution_17, unsqueeze_618, buf293, 1568, 128, grid=grid(1568), stream=stream0)
        buf294 = empty((32, ), device='cuda', dtype=torch.float32)
        buf295 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_65.run(buf293, squeeze_52, buf294, buf295, 32, 49, grid=grid(32), stream=stream0)
        del buf293
        buf296 = empty((8, 32, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_66.run(buf290, convolution_17, unsqueeze_618, buf294, squeeze_52, buf292, primals_35, buf296, 256, 784, grid=grid(256, 784), stream=stream0)
        del convolution_17
        del primals_35
        del squeeze_52
        del unsqueeze_618
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf297 = aten.convolution_backward(buf296, clamp_max_11, primals_122, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clamp_max_11
        del primals_122
        buf298 = buf297[0]
        buf299 = buf297[1]
        del buf297
        buf300 = reinterpret_tensor(buf285, (192, 49), (49, 1), 0); del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_58.run(bitwise_or_23, buf298, buf300, 9408, 128, grid=grid(9408), stream=stream0)
        buf301 = buf286; del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_59.run(buf300, buf301, 192, 49, grid=grid(192), stream=stream0)
        buf302 = reinterpret_tensor(buf300, (192, 49), (1, 192), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_60.run(bitwise_or_23, buf298, convolution_16, unsqueeze_630, buf302, 9408, 128, grid=grid(9408), stream=stream0)
        buf303 = empty((192, ), device='cuda', dtype=torch.float32)
        buf304 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_61.run(buf302, squeeze_49, buf303, buf304, 192, 49, grid=grid(192), stream=stream0)
        buf305 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_62.run(bitwise_or_23, buf298, convolution_16, unsqueeze_630, buf303, squeeze_49, buf301, primals_33, buf305, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del bitwise_or_23
        del buf298
        del convolution_16
        del primals_33
        del squeeze_49
        del unsqueeze_630
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf306 = aten.convolution_backward(buf305, clamp_max_10, primals_121, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False])
        del clamp_max_10
        del primals_121
        buf307 = buf306[0]
        buf308 = buf306[1]
        del buf306
        buf309 = reinterpret_tensor(buf302, (192, 49), (49, 1), 0); del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_58.run(bitwise_or_24, buf307, buf309, 9408, 128, grid=grid(9408), stream=stream0)
        buf310 = buf303; del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_59.run(buf309, buf310, 192, 49, grid=grid(192), stream=stream0)
        buf311 = reinterpret_tensor(buf309, (192, 49), (1, 192), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_60.run(bitwise_or_24, buf307, convolution_15, unsqueeze_642, buf311, 9408, 128, grid=grid(9408), stream=stream0)
        buf312 = empty((192, ), device='cuda', dtype=torch.float32)
        buf313 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_61.run(buf311, squeeze_46, buf312, buf313, 192, 49, grid=grid(192), stream=stream0)
        buf314 = buf305; del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_62.run(bitwise_or_24, buf307, convolution_15, unsqueeze_642, buf312, squeeze_46, buf310, primals_31, buf314, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del bitwise_or_24
        del buf307
        del convolution_15
        del primals_31
        del squeeze_46
        del unsqueeze_642
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf315 = aten.convolution_backward(buf314, add_76, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_76
        del primals_120
        buf316 = buf315[0]
        buf317 = buf315[1]
        del buf315
        buf318 = buf294; del buf294  # reuse
        buf319 = empty((32, ), device='cuda', dtype=torch.float32)
        buf320 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_67.run(buf290, buf316, convolution_14, unsqueeze_654, squeeze_43, buf318, buf319, buf320, 32, 6272, grid=grid(32), stream=stream0)
        buf321 = buf296; del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_68.run(buf290, buf316, convolution_14, unsqueeze_654, buf319, squeeze_43, buf318, primals_29, buf321, 256, 784, grid=grid(256, 784), stream=stream0)
        del convolution_14
        del primals_29
        del squeeze_43
        del unsqueeze_654
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf322 = aten.convolution_backward(buf321, clamp_max_9, primals_119, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf321
        del clamp_max_9
        del primals_119
        buf323 = buf322[0]
        buf324 = buf322[1]
        del buf322
        buf325 = reinterpret_tensor(buf311, (192, 49), (49, 1), 0); del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_58.run(bitwise_or_25, buf323, buf325, 9408, 128, grid=grid(9408), stream=stream0)
        buf326 = buf312; del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_59.run(buf325, buf326, 192, 49, grid=grid(192), stream=stream0)
        buf327 = reinterpret_tensor(buf325, (192, 49), (1, 192), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_60.run(bitwise_or_25, buf323, convolution_13, unsqueeze_666, buf327, 9408, 128, grid=grid(9408), stream=stream0)
        buf328 = empty((192, ), device='cuda', dtype=torch.float32)
        buf329 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_61.run(buf327, squeeze_40, buf328, buf329, 192, 49, grid=grid(192), stream=stream0)
        buf330 = buf314; del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_62.run(bitwise_or_25, buf323, convolution_13, unsqueeze_666, buf328, squeeze_40, buf326, primals_27, buf330, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del bitwise_or_25
        del buf323
        del convolution_13
        del primals_27
        del squeeze_40
        del unsqueeze_666
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf331 = aten.convolution_backward(buf330, clamp_max_8, primals_118, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False])
        del clamp_max_8
        del primals_118
        buf332 = buf331[0]
        buf333 = buf331[1]
        del buf331
        buf334 = reinterpret_tensor(buf327, (192, 49), (49, 1), 0); del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_58.run(bitwise_or_26, buf332, buf334, 9408, 128, grid=grid(9408), stream=stream0)
        buf335 = buf328; del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_59.run(buf334, buf335, 192, 49, grid=grid(192), stream=stream0)
        buf336 = reinterpret_tensor(buf334, (192, 49), (1, 192), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_60.run(bitwise_or_26, buf332, convolution_12, unsqueeze_678, buf336, 9408, 128, grid=grid(9408), stream=stream0)
        buf337 = empty((192, ), device='cuda', dtype=torch.float32)
        buf338 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_61.run(buf336, squeeze_37, buf337, buf338, 192, 49, grid=grid(192), stream=stream0)
        del buf336
        buf339 = buf330; del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_62.run(bitwise_or_26, buf332, convolution_12, unsqueeze_678, buf337, squeeze_37, buf335, primals_25, buf339, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del bitwise_or_26
        del buf332
        del buf337
        del convolution_12
        del primals_25
        del squeeze_37
        del unsqueeze_678
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf340 = aten.convolution_backward(buf339, add_60, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_60
        del buf339
        del primals_117
        buf341 = buf340[0]
        buf342 = buf340[1]
        del buf340
        buf343 = buf319; del buf319  # reuse
        buf344 = empty((32, ), device='cuda', dtype=torch.float32)
        buf346 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_69.run(buf290, buf316, buf341, convolution_11, unsqueeze_690, squeeze_34, buf343, buf344, buf346, 32, 6272, grid=grid(32), stream=stream0)
        buf345 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_70.run(buf345, buf316, buf341, convolution_11, unsqueeze_690, buf344, squeeze_34, buf343, primals_23, 256, 784, grid=grid(256, 784), stream=stream0)
        del buf316
        del buf341
        del convolution_11
        del primals_23
        del squeeze_34
        del unsqueeze_690
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf347 = aten.convolution_backward(buf345, clamp_max_7, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf345
        del clamp_max_7
        del primals_116
        buf348 = buf347[0]
        buf349 = buf347[1]
        del buf347
        buf350 = empty((144, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_71.run(bitwise_or_27, buf348, buf350, 7056, 128, grid=grid(7056), stream=stream0)
        buf351 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_72.run(buf350, buf351, 144, 49, grid=grid(144), stream=stream0)
        buf352 = reinterpret_tensor(buf350, (144, 49), (1, 144), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_73.run(bitwise_or_27, buf348, convolution_10, unsqueeze_702, buf352, 7056, 128, grid=grid(7056), stream=stream0)
        buf353 = empty((144, ), device='cuda', dtype=torch.float32)
        buf354 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_74.run(buf352, squeeze_31, buf353, buf354, 144, 49, grid=grid(144), stream=stream0)
        del buf352
        buf355 = reinterpret_tensor(buf161, (8, 144, 28, 28), (112896, 1, 4032, 144), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_75.run(bitwise_or_27, buf348, convolution_10, unsqueeze_702, buf353, squeeze_31, buf351, primals_21, buf355, 6272, 144, grid=grid(6272, 144), stream=stream0)
        del bitwise_or_27
        del buf348
        del convolution_10
        del primals_21
        del squeeze_31
        del unsqueeze_702
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf356 = aten.convolution_backward(buf355, clamp_max_6, primals_115, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False])
        del buf355
        del clamp_max_6
        del primals_115
        buf357 = buf356[0]
        buf358 = buf356[1]
        del buf356
        buf359 = empty((144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_76.run(bitwise_or_28, buf357, buf359, 28224, 128, grid=grid(28224), stream=stream0)
        buf360 = buf353; del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_77.run(buf359, buf360, 144, 196, grid=grid(144), stream=stream0)
        buf361 = reinterpret_tensor(buf359, (144, 196), (1, 144), 0); del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_78.run(bitwise_or_28, buf357, convolution_9, unsqueeze_714, buf361, 28224, 128, grid=grid(28224), stream=stream0)
        buf362 = empty((144, ), device='cuda', dtype=torch.float32)
        buf363 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_79.run(buf361, squeeze_28, buf362, buf363, 144, 196, grid=grid(144), stream=stream0)
        buf364 = empty_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_80.run(bitwise_or_28, buf357, convolution_9, unsqueeze_714, buf362, squeeze_28, buf360, primals_19, buf364, 25088, 144, grid=grid(25088, 144), stream=stream0)
        del bitwise_or_28
        del buf357
        del convolution_9
        del primals_19
        del squeeze_28
        del unsqueeze_714
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf365 = aten.convolution_backward(buf364, add_45, primals_114, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_45
        del primals_114
        buf366 = buf365[0]
        buf367 = buf365[1]
        del buf365
        buf368 = reinterpret_tensor(buf166, (24, 4), (1, 24), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf366, buf368, 96, 6272, grid=grid(96), stream=stream0)
        buf369 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_82.run(buf368, buf369, 24, 4, grid=grid(24), stream=stream0)
        buf370 = empty((24, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_83.run(buf366, convolution_8, unsqueeze_726, buf370, 4704, 128, grid=grid(4704), stream=stream0)
        buf371 = empty((24, ), device='cuda', dtype=torch.float32)
        buf372 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_84.run(buf370, squeeze_25, buf371, buf372, 24, 196, grid=grid(24), stream=stream0)
        del buf370
        buf373 = reinterpret_tensor(buf262, (8, 24, 56, 56), (75264, 3136, 56, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_85.run(buf366, convolution_8, unsqueeze_726, buf371, squeeze_25, buf369, primals_17, buf373, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del convolution_8
        del primals_17
        del squeeze_25
        del unsqueeze_726
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf374 = aten.convolution_backward(buf373, clamp_max_5, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf373
        del clamp_max_5
        del primals_113
        buf375 = buf374[0]
        buf376 = buf374[1]
        del buf374
        buf377 = reinterpret_tensor(buf361, (144, 196), (196, 1), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_76.run(bitwise_or_29, buf375, buf377, 28224, 128, grid=grid(28224), stream=stream0)
        buf378 = buf362; del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_77.run(buf377, buf378, 144, 196, grid=grid(144), stream=stream0)
        buf379 = reinterpret_tensor(buf377, (144, 196), (1, 144), 0); del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_78.run(bitwise_or_29, buf375, convolution_7, unsqueeze_738, buf379, 28224, 128, grid=grid(28224), stream=stream0)
        buf380 = empty((144, ), device='cuda', dtype=torch.float32)
        buf381 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_79.run(buf379, squeeze_22, buf380, buf381, 144, 196, grid=grid(144), stream=stream0)
        buf382 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_80.run(bitwise_or_29, buf375, convolution_7, unsqueeze_738, buf380, squeeze_22, buf378, primals_15, buf382, 25088, 144, grid=grid(25088, 144), stream=stream0)
        del bitwise_or_29
        del buf375
        del convolution_7
        del primals_15
        del squeeze_22
        del unsqueeze_738
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf383 = aten.convolution_backward(buf382, clamp_max_4, primals_112, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False])
        del clamp_max_4
        del primals_112
        buf384 = buf383[0]
        buf385 = buf383[1]
        del buf383
        buf386 = reinterpret_tensor(buf379, (144, 196), (196, 1), 0); del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_76.run(bitwise_or_30, buf384, buf386, 28224, 128, grid=grid(28224), stream=stream0)
        buf387 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_77.run(buf386, buf387, 144, 196, grid=grid(144), stream=stream0)
        buf388 = reinterpret_tensor(buf386, (144, 196), (1, 144), 0); del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_78.run(bitwise_or_30, buf384, convolution_6, unsqueeze_750, buf388, 28224, 128, grid=grid(28224), stream=stream0)
        buf389 = empty((144, ), device='cuda', dtype=torch.float32)
        buf390 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_79.run(buf388, squeeze_19, buf389, buf390, 144, 196, grid=grid(144), stream=stream0)
        del buf388
        buf391 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_80.run(bitwise_or_30, buf384, convolution_6, unsqueeze_750, buf389, squeeze_19, buf387, primals_13, buf391, 25088, 144, grid=grid(25088, 144), stream=stream0)
        del bitwise_or_30
        del buf384
        del buf389
        del convolution_6
        del primals_13
        del squeeze_19
        del unsqueeze_750
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf392 = aten.convolution_backward(buf391, add_29, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_29
        del buf391
        del primals_111
        buf393 = buf392[0]
        buf394 = buf392[1]
        del buf392
        buf395 = buf368; del buf368  # reuse
        buf397 = empty_strided((24, 4), (1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_86.run(buf366, buf393, convolution_5, unsqueeze_762, buf395, buf397, 96, 6272, grid=grid(96), stream=stream0)
        buf396 = buf371; del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_82.run(buf395, buf396, 24, 4, grid=grid(24), stream=stream0)
        buf398 = empty((24, ), device='cuda', dtype=torch.float32)
        buf399 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_87.run(buf397, squeeze_16, buf398, buf399, 24, 4, grid=grid(24), stream=stream0)
        buf400 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_88.run(buf400, buf393, convolution_5, unsqueeze_762, buf398, squeeze_16, buf396, primals_11, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del buf393
        del buf398
        del convolution_5
        del primals_11
        del squeeze_16
        del unsqueeze_762
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf401 = aten.convolution_backward(buf400, clamp_max_3, primals_110, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf400
        del clamp_max_3
        del primals_110
        buf402 = buf401[0]
        buf403 = buf401[1]
        del buf401
        buf404 = empty((96, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_89.run(bitwise_or_31, buf402, buf404, 18816, 128, grid=grid(18816), stream=stream0)
        buf405 = reinterpret_tensor(buf397, (96, ), (1, ), 0); del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_90.run(buf404, buf405, 96, 196, grid=grid(96), stream=stream0)
        buf406 = reinterpret_tensor(buf404, (96, 196), (1, 96), 0); del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_91.run(bitwise_or_31, buf402, convolution_4, unsqueeze_774, buf406, 18816, 128, grid=grid(18816), stream=stream0)
        buf407 = reinterpret_tensor(buf395, (96, ), (1, ), 0); del buf395  # reuse
        buf408 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_92.run(buf406, squeeze_13, buf407, buf408, 96, 196, grid=grid(96), stream=stream0)
        del buf406
        buf409 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_93.run(bitwise_or_31, buf402, convolution_4, unsqueeze_774, buf407, squeeze_13, buf405, primals_9, buf409, 25088, 96, grid=grid(25088, 96), stream=stream0)
        del bitwise_or_31
        del buf402
        del convolution_4
        del primals_9
        del squeeze_13
        del unsqueeze_774
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf410 = aten.convolution_backward(buf409, clamp_max_2, primals_109, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 96, [True, True, False])
        del buf409
        del clamp_max_2
        del primals_109
        buf411 = buf410[0]
        buf412 = buf410[1]
        del buf410
        buf413 = empty((96, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_94.run(bitwise_or_32, buf411, buf413, 75264, 128, grid=grid(75264), stream=stream0)
        buf414 = buf407; del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_95.run(buf413, buf414, 96, 784, grid=grid(96), stream=stream0)
        buf415 = reinterpret_tensor(buf413, (96, 784), (1, 96), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_96.run(bitwise_or_32, buf411, convolution_3, unsqueeze_786, buf415, 75264, 128, grid=grid(75264), stream=stream0)
        buf416 = empty((96, ), device='cuda', dtype=torch.float32)
        buf417 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_97.run(buf415, squeeze_10, buf416, buf417, 96, 784, grid=grid(96), stream=stream0)
        del buf415
        buf418 = empty_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_98.run(bitwise_or_32, buf411, convolution_3, unsqueeze_786, buf416, squeeze_10, buf414, primals_7, buf418, 100352, 96, grid=grid(100352, 96), stream=stream0)
        del bitwise_or_32
        del buf411
        del buf416
        del convolution_3
        del primals_7
        del squeeze_10
        del unsqueeze_786
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf419 = aten.convolution_backward(buf418, add_14, primals_108, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_14
        del buf418
        del primals_108
        buf420 = buf419[0]
        buf421 = buf419[1]
        del buf419
        buf422 = empty((16, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_99.run(buf420, buf422, 208, 7720, grid=grid(208), stream=stream0)
        buf423 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_100.run(buf422, buf423, 16, 13, grid=grid(16), stream=stream0)
        del buf422
        buf424 = empty((16, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_101.run(buf420, convolution_2, unsqueeze_798, buf424, 12544, 128, grid=grid(12544), stream=stream0)
        buf425 = empty((16, ), device='cuda', dtype=torch.float32)
        buf426 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_102.run(buf424, squeeze_7, buf425, buf426, 16, 784, grid=grid(16), stream=stream0)
        del buf424
        buf427 = buf420; del buf420  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_103.run(buf427, convolution_2, unsqueeze_798, buf425, squeeze_7, buf423, primals_5, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del buf425
        del convolution_2
        del primals_5
        del squeeze_7
        del unsqueeze_798
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf428 = aten.convolution_backward(buf427, clamp_max_1, primals_107, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf427
        del clamp_max_1
        del primals_107
        buf429 = buf428[0]
        buf430 = buf428[1]
        del buf428
        buf431 = empty((32, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_104.run(bitwise_or_33, buf429, buf431, 25088, 128, grid=grid(25088), stream=stream0)
        buf432 = buf344; del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_105.run(buf431, buf432, 32, 784, grid=grid(32), stream=stream0)
        buf433 = reinterpret_tensor(buf431, (32, 784), (1, 32), 0); del buf431  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_106.run(bitwise_or_33, buf429, convolution_1, unsqueeze_810, buf433, 25088, 128, grid=grid(25088), stream=stream0)
        buf434 = empty((32, ), device='cuda', dtype=torch.float32)
        buf435 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_107.run(buf433, squeeze_4, buf434, buf435, 32, 784, grid=grid(32), stream=stream0)
        buf436 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_108.run(bitwise_or_33, buf429, convolution_1, unsqueeze_810, buf434, squeeze_4, buf432, primals_3, buf436, 100352, 32, grid=grid(100352, 32), stream=stream0)
        del bitwise_or_33
        del buf429
        del convolution_1
        del primals_3
        del squeeze_4
        del unsqueeze_810
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf437 = aten.convolution_backward(buf436, clamp_max, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del clamp_max
        del primals_106
        buf438 = buf437[0]
        buf439 = buf437[1]
        del buf437
        buf440 = reinterpret_tensor(buf433, (32, 784), (784, 1), 0); del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_104.run(bitwise_or_34, buf438, buf440, 25088, 128, grid=grid(25088), stream=stream0)
        buf441 = buf434; del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_105.run(buf440, buf441, 32, 784, grid=grid(32), stream=stream0)
        buf442 = reinterpret_tensor(buf440, (32, 784), (1, 32), 0); del buf440  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_106.run(bitwise_or_34, buf438, convolution, unsqueeze_822, buf442, 25088, 128, grid=grid(25088), stream=stream0)
        buf443 = empty((32, ), device='cuda', dtype=torch.float32)
        buf444 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_107.run(buf442, squeeze_1, buf443, buf444, 32, 784, grid=grid(32), stream=stream0)
        del buf442
        buf445 = buf436; del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_108.run(bitwise_or_34, buf438, convolution, unsqueeze_822, buf443, squeeze_1, buf441, primals_1, buf445, 100352, 32, grid=grid(100352, 32), stream=stream0)
        del bitwise_or_34
        del buf438
        del buf443
        del convolution
        del primals_1
        del squeeze_1
        del unsqueeze_822
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf446 = aten.convolution_backward(buf445, primals_315, primals_105, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf445
        del primals_105
        del primals_315
        buf447 = buf446[1]
        return (buf444, buf441, buf435, buf432, buf426, buf423, buf417, buf414, buf408, buf405, buf399, buf396, buf390, buf387, buf381, buf378, buf372, buf369, buf363, buf360, buf354, buf351, buf346, buf343, buf338, buf335, buf329, buf326, buf320, buf318, buf313, buf310, buf304, buf301, buf295, buf292, buf287, buf284, buf278, buf275, buf269, buf266, buf261, buf258, buf252, buf249, buf244, buf241, buf236, buf233, buf227, buf224, buf218, buf216, buf211, buf208, buf202, buf199, buf193, buf190, buf185, buf182, buf176, buf173, buf168, buf165, buf160, buf157, buf151, buf148, buf142, buf140, buf135, buf132, buf126, buf123, buf117, buf114, buf109, buf106, buf100, buf97, buf92, buf89, buf84, buf81, buf75, buf72, buf66, buf64, buf59, buf56, buf50, buf47, buf41, buf38, buf33, buf30, buf24, buf21, buf15, buf12, buf7, buf4, buf447, buf439, buf430, buf421, buf412, buf403, buf394, buf385, buf376, buf367, buf358, buf349, buf342, buf333, buf324, buf317, buf308, buf299, buf291, buf282, buf273, buf265, buf256, buf247, buf240, buf231, buf222, buf215, buf206, buf197, buf189, buf180, buf171, buf164, buf155, buf146, buf139, buf130, buf121, buf113, buf104, buf95, buf88, buf79, buf70, buf63, buf54, buf45, buf37, buf28, buf19, buf11, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((32, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((160, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((320, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_1 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_14 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_2 = rand_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_3 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_29 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_4 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_5 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_45 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_6 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_7 = rand_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_60 = rand_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_8 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_9 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_76 = rand_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_10 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_11 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_92 = rand_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_12 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_13 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_107 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_14 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_15 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_123 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_16 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_17 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_139 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_18 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_19 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_155 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_20 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_21 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_170 = rand_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_22 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_23 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_186 = rand_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_24 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_25 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_202 = rand_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_26 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 576, 7, 7), (28224, 1, 4032, 576), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_27 = rand_strided((8, 576, 7, 7), (28224, 1, 4032, 576), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_217 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_28 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_29 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_233 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_30 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_31 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_249 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_32 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_33 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 320, 7, 7), (15680, 1, 2240, 320), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_264 = rand_strided((8, 320, 7, 7), (15680, 1, 2240, 320), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((8, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or = rand_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda:0', dtype=torch.bool)
    unsqueeze_210 = rand_strided((1, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_222 = rand_strided((1, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_1 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.bool)
    unsqueeze_234 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_2 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.bool)
    unsqueeze_246 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_3 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.bool)
    unsqueeze_270 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_4 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.bool)
    unsqueeze_282 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_294 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_5 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.bool)
    unsqueeze_306 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_6 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.bool)
    unsqueeze_318 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_7 = rand_strided((8, 576, 7, 7), (28224, 1, 4032, 576), device='cuda:0', dtype=torch.bool)
    unsqueeze_342 = rand_strided((1, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_8 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.bool)
    unsqueeze_354 = rand_strided((1, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_366 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_9 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.bool)
    unsqueeze_378 = rand_strided((1, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_10 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.bool)
    unsqueeze_390 = rand_strided((1, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_11 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.bool)
    unsqueeze_414 = rand_strided((1, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_12 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda:0', dtype=torch.bool)
    unsqueeze_426 = rand_strided((1, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_438 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_13 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_450 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_14 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_462 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_15 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_486 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_16 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_498 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_510 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_17 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_522 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_18 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_534 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_546 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_19 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_558 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_20 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.bool)
    unsqueeze_570 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_582 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_21 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_594 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_22 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_606 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_618 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_23 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_630 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_24 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_642 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_25 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_666 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_26 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_678 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_690 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_27 = rand_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cuda:0', dtype=torch.bool)
    unsqueeze_702 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_28 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.bool)
    unsqueeze_714 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_726 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_29 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.bool)
    unsqueeze_738 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_30 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda:0', dtype=torch.bool)
    unsqueeze_750 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_762 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_31 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.bool)
    unsqueeze_774 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_32 = rand_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda:0', dtype=torch.bool)
    unsqueeze_786 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_798 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_33 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.bool)
    unsqueeze_810 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_34 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.bool)
    unsqueeze_822 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_315, convolution, squeeze_1, clamp_max, convolution_1, squeeze_4, clamp_max_1, convolution_2, squeeze_7, add_14, convolution_3, squeeze_10, clamp_max_2, convolution_4, squeeze_13, clamp_max_3, convolution_5, squeeze_16, add_29, convolution_6, squeeze_19, clamp_max_4, convolution_7, squeeze_22, clamp_max_5, convolution_8, squeeze_25, add_45, convolution_9, squeeze_28, clamp_max_6, convolution_10, squeeze_31, clamp_max_7, convolution_11, squeeze_34, add_60, convolution_12, squeeze_37, clamp_max_8, convolution_13, squeeze_40, clamp_max_9, convolution_14, squeeze_43, add_76, convolution_15, squeeze_46, clamp_max_10, convolution_16, squeeze_49, clamp_max_11, convolution_17, squeeze_52, add_92, convolution_18, squeeze_55, clamp_max_12, convolution_19, squeeze_58, clamp_max_13, convolution_20, squeeze_61, add_107, convolution_21, squeeze_64, clamp_max_14, convolution_22, squeeze_67, clamp_max_15, convolution_23, squeeze_70, add_123, convolution_24, squeeze_73, clamp_max_16, convolution_25, squeeze_76, clamp_max_17, convolution_26, squeeze_79, add_139, convolution_27, squeeze_82, clamp_max_18, convolution_28, squeeze_85, clamp_max_19, convolution_29, squeeze_88, add_155, convolution_30, squeeze_91, clamp_max_20, convolution_31, squeeze_94, clamp_max_21, convolution_32, squeeze_97, add_170, convolution_33, squeeze_100, clamp_max_22, convolution_34, squeeze_103, clamp_max_23, convolution_35, squeeze_106, add_186, convolution_36, squeeze_109, clamp_max_24, convolution_37, squeeze_112, clamp_max_25, convolution_38, squeeze_115, add_202, convolution_39, squeeze_118, clamp_max_26, convolution_40, squeeze_121, clamp_max_27, convolution_41, squeeze_124, add_217, convolution_42, squeeze_127, clamp_max_28, convolution_43, squeeze_130, clamp_max_29, convolution_44, squeeze_133, add_233, convolution_45, squeeze_136, clamp_max_30, convolution_46, squeeze_139, clamp_max_31, convolution_47, squeeze_142, add_249, convolution_48, squeeze_145, clamp_max_32, convolution_49, squeeze_148, clamp_max_33, convolution_50, squeeze_151, add_264, convolution_51, squeeze_154, view, permute_1, bitwise_or, unsqueeze_210, unsqueeze_222, bitwise_or_1, unsqueeze_234, bitwise_or_2, unsqueeze_246, unsqueeze_258, bitwise_or_3, unsqueeze_270, bitwise_or_4, unsqueeze_282, unsqueeze_294, bitwise_or_5, unsqueeze_306, bitwise_or_6, unsqueeze_318, unsqueeze_330, bitwise_or_7, unsqueeze_342, bitwise_or_8, unsqueeze_354, unsqueeze_366, bitwise_or_9, unsqueeze_378, bitwise_or_10, unsqueeze_390, unsqueeze_402, bitwise_or_11, unsqueeze_414, bitwise_or_12, unsqueeze_426, unsqueeze_438, bitwise_or_13, unsqueeze_450, bitwise_or_14, unsqueeze_462, unsqueeze_474, bitwise_or_15, unsqueeze_486, bitwise_or_16, unsqueeze_498, unsqueeze_510, bitwise_or_17, unsqueeze_522, bitwise_or_18, unsqueeze_534, unsqueeze_546, bitwise_or_19, unsqueeze_558, bitwise_or_20, unsqueeze_570, unsqueeze_582, bitwise_or_21, unsqueeze_594, bitwise_or_22, unsqueeze_606, unsqueeze_618, bitwise_or_23, unsqueeze_630, bitwise_or_24, unsqueeze_642, unsqueeze_654, bitwise_or_25, unsqueeze_666, bitwise_or_26, unsqueeze_678, unsqueeze_690, bitwise_or_27, unsqueeze_702, bitwise_or_28, unsqueeze_714, unsqueeze_726, bitwise_or_29, unsqueeze_738, bitwise_or_30, unsqueeze_750, unsqueeze_762, bitwise_or_31, unsqueeze_774, bitwise_or_32, unsqueeze_786, unsqueeze_798, bitwise_or_33, unsqueeze_810, bitwise_or_34, unsqueeze_822, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilenetv2_100', benchmark_compiled_module)
