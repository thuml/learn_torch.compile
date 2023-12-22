
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


# kernel path: /tmp/torchinductor_youkaichao/ue/cuea7xzgrpl7q7q4un2mkpv5rilgsblwvi4xsjcmezhsr2rajrkw.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (200704*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (x0 + (2048*(r2 // 49)) + (4096*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (2048*r2) + (200704*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/mf/cmf5t4e6xvm27nrgloapcstyjquqh3s4cbjtoiat2vpmbe6wdqsr.py
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
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/k5/ck5xfolx5trg253cosfgkr7lq2zxghfuag74lhbxh4xes2wmutnz.py
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
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r1)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, None)
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ex/cex74hvhpmvea2l3hcxqmbclqrk5qz4mytdyhqp7lvzenxmdvqir.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 2048
    x2 = (xindex // 100352)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (2048*x2)), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/4p/c4pzxj4eqpu5chm3fpeddxfkmo335og6qdsa3e2cy42yxkpkfj5k.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 208
    x1 = (xindex // 208)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (208*r2) + (20384*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (20384 + (49*x0) + (40768*(r2 // 49)) + (81536*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (208*r2) + (20384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/f4/cf4bkzbonwxwsho3z57rhoxwx5aaalbyecj4pljeklufvgh7ivwj.py
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
    size_hints=[256, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 208
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (208*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g3/cg32pkfmwws4lwo3gaxzf5hnstlzf52xi2jy5zyyg3aidcnobc5a.py
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
    size_hints=[256, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 208
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (208*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ys/cysy4u37no7lxtaaai7dqksjkgzje3ru4ylv7oh42y7obsuouxce.py
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
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 208
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
    tmp0 = tl.load(in_ptr0 + (x2 + (208*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (20384 + y0 + (49*x2) + (40768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (208*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (208*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6r/c6reg4w5itezauc5uu46doagtxqmg5zaouwgh6dzltjmp6g53t3d.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
    xnumel = 208
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
    tmp0 = tl.load(in_ptr0 + (x0 + (208*r3)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (10192 + r1 + (49*x0) + (40768*r2)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (49*x0) + (10192*r2)), rmask & xmask, other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pc/cpcwdkdj2nsiqcffm2cqyiven5w4velojnzjoemf3tziw4gdavpw.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 208
    x1 = (xindex // 208)
    tmp7 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (208*r2) + (20384*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (10192 + (49*x0) + (40768*(r2 // 49)) + (81536*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + ((49*x0) + (10192*(r2 // 49)) + (20384*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x0 + (208*r2) + (20384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ug/cuglvjalskjwohbfrz66rvsxxfrl2bo4rwrad3kjmyew4aoucugz.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 208
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
    tmp0 = tl.load(in_ptr0 + (x2 + (208*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (10192 + y0 + (49*x2) + (40768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (49*x2) + (10192*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (208*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
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
    tl.store(out_ptr0 + (x2 + (208*y3)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lz/clzbwtudgapvcshy7hgyimccfo7rq2li3jedxcjdj7wr7rfelkt6.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
    xnumel = 208
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
    tmp0 = tl.load(in_ptr0 + (x0 + (208*r3)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (40768*r2)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (49*x0) + (10192*r2)), rmask & xmask, other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5t/c5tubd3k3ozy65hy446wxnja2ktnxgng2bmn2gpeda2lsrrkk4e6.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 208
    x1 = (xindex // 208)
    tmp7 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (208*r2) + (20384*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((49*x0) + (40768*(r2 // 49)) + (81536*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + ((49*x0) + (10192*(r2 // 49)) + (20384*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x0 + (208*r2) + (20384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/lo/clovc2flxj5glxavlocc6ae2vjq7ztc57c6g2k2cxfltgk2gsqao.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 208
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
    tmp0 = tl.load(in_ptr0 + (x2 + (208*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (49*x2) + (40768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (49*x2) + (10192*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (208*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
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
    tl.store(out_ptr0 + (x2 + (208*y3)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cb/ccbibknhydj6swyrz7jmmcarpxq5miea2ftoyz3y5scsh7s7h5mu.py
# Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]

triton_poi_fused_cat_threshold_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_threshold_backward_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6656
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 832
    y1 = (yindex // 832)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (832*x2) + (40768*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 208, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (49*y0) + (10192*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 416, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-10192) + x2 + (49*y0) + (10192*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1, 1], 624, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + ((-20384) + x2 + (49*y0) + (10192*y1)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp1 >= tmp17
    tmp24 = tl.full([1, 1], 832, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.load(in_out_ptr0 + (x2 + (49*y3)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp12, tmp15, tmp29)
    tmp31 = tl.where(tmp5, tmp8, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp0, tmp32, tmp31)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp33, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/43/c43rp5wonbyr2u4lujddjho4mxhq23r3kft4dqjnhcbnn3rwwni6.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 832
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (40768*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bb/cbbucl2by5lma3ydqe5oeqcp5vjqkblvtdidk6bwes4czxh66wwu.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 832
    x1 = (xindex // 832)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (40768*(r2 // 49)) + (81536*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (832*r2) + (81536*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6w/c6wsmakia5bsvkjnfvaoggkm4cb35wtr2qvlfh6n6tzjvrb6phz3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (832*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yc/cycwmd5pxebcrzc2umukbidgnytdbvzlnauiu6dw7uppmnnc4x4r.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 832
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (40768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (832*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (832*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iv/civyqk27zfmsm4xxombyt37icghb72n6w4qrquwpbckgh5evk7w3.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_native_batch_norm_backward_threshold_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_native_batch_norm_backward_threshold_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (200704*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (2048*r2) + (200704*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp4 = tl.load(in_ptr2 + (x0 + (2048*(r2 // 49)) + (4096*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + ((49*x0) + (100352*(r2 // 49)) + (200704*x1) + (r2 % 49)), rmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr4 + (x0 + (2048*r2) + (200704*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = 49.0
        tmp6 = tmp4 / tmp5
        tmp7 = tl.where(tmp3, tmp1, tmp6)
        tmp9 = tmp7 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gk/cgkgouo6eotbm2akg26335uxwztzadtiszcm3v7simiybasycvhk.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 49)
    y0 = yindex % 49
    tmp0 = tl.load(in_ptr0 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last').to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (x2 + (2048*y1)), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0 + (49*x2) + (100352*y1)), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr9 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tmp7 = tl.where(tmp3, tmp1, tmp6)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.002551020408163265
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
    tl.store(in_out_ptr0 + (x2 + (2048*y3)), tmp27, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6i/c6iumtyq3kpdounimq5lq7svddloc4vzfom5ecesejcf55uewloe.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.threshold_backward]

triton_poi_fused_add_div_threshold_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_threshold_backward_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 49)
    y0 = yindex % 49
    tmp0 = tl.load(in_ptr0 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last').to(tl.int1)
    tmp6 = tl.load(in_ptr3 + (x2 + (2048*y1)), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (y0 + (49*x2) + (100352*y1)), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (y0 + (49*x2) + (100352*y1)), ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tmp9 = tl.where(tmp5, tmp1, tmp8)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp4, tmp1, tmp11)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tl.store(out_ptr0 + (x2 + (2048*y3)), tmp15, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ay/caygy5r4dvejtuzupvwkdu7xr4capy23h545hfbxqp44ozepx33c.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (200704*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (2048*r2) + (200704*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr3 + (x0 + (2048*r2) + (200704*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp0 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, None)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mg/cmgs6zahfxefzc2nmzlik7maftydld2vmlfnxvrfz3xxg4wcaroh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), None)
    tmp19 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
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
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x2), tmp17, None)
    tl.store(out_ptr1 + (x2), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tl/ctl3i6r4eyjiipvdcjexsvr224hjtiembnpb3n32cv2no2x53ogg.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 326144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x1 = (xindex // 14) % 14
    x2 = (xindex // 196) % 208
    x3 = (xindex // 40768)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (30576 + (7*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2))))) >= 0, 0, 7))) + (49*x2) + (40768*x3) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) >= 0, 0, 7))), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (30576 + (7*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2))))) >= 0, 0, 7))) + (49*x2) + (40768*x3) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) >= 0, 0, 7))), xmask)
    tmp18 = tl.load(in_ptr0 + (30576 + (7*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2)))))) + (7*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2))))) >= 0, 0, 7))) + (49*x2) + (40768*x3) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) >= 0, 0, 7))), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (30576 + (7*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2)))))) + (7*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x1) // 2))))) >= 0, 0, 7))) + (49*x2) + (40768*x3) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(7, 1 + ((1 + x0) // 2))))) >= 0, 0, 7))), xmask)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (x1 // 2))
    tmp3 = tl.math.min(7, 1 + ((1 + x1) // 2))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (x0 // 2))
    tmp6 = tl.math.min(7, 1 + ((1 + x0) // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.math.max(0, (x0 // 2)))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 1 + (tl.math.max(0, (x1 // 2)))
    tmp21 = tmp20 < tmp3
    tmp22 = tmp21 & tmp7
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = tmp21 & tmp14
    tmp28 = tmp24 + tmp26
    tmp29 = tl.where(tmp27, tmp28, tmp24)
    tl.store(out_ptr0 + (x6), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n7/cn75m7g243vbgxndlmx2qlzsbgq5ljbcwen2kqqkocxl7yjcvbz3.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 208
    x1 = (xindex // 208)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (208*r2) + (20384*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (10192 + (49*x0) + (40768*(r2 // 49)) + (81536*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (208*r2) + (20384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7k/c7knzeo5avebs2nuaibin4a5smta6i2jgiubkacogz44ap5utbkc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 208
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
    tmp0 = tl.load(in_ptr0 + (x2 + (208*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (10192 + y0 + (49*x2) + (40768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (208*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (208*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/su/csuqk462tv4terzrvwaldy3ryogmtefy7lriv2jn5z3wonbkeql3.py
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 208
    x1 = (xindex // 208)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (208*r2) + (20384*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((49*x0) + (40768*(r2 // 49)) + (81536*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (208*r2) + (20384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qy/cqyy6t2aw6cypnzncsv3vluoktcwvte5uturyn556svgkjqrvrvf.py
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
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 208
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
    tmp0 = tl.load(in_ptr0 + (x2 + (208*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (49*x2) + (40768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (208*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (208*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jy/cjyywcbsrupo6p3zfshlzl5rko63uthyxyahpqnwheuoekcvbumg.py
# Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]

triton_poi_fused_cat_threshold_backward_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_threshold_backward_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6656
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 832
    y1 = (yindex // 832)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (832*x2) + (163072*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 208, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (196*y0) + (40768*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 416, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-40768) + x2 + (196*y0) + (40768*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1, 1], 624, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + ((-81536) + x2 + (196*y0) + (40768*y1)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp1 >= tmp17
    tmp24 = tl.full([1, 1], 832, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.load(in_ptr4 + ((-122304) + x2 + (196*y0) + (40768*y1)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp12, tmp15, tmp29)
    tmp31 = tl.where(tmp5, tmp8, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp0, tmp32, tmp31)
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp33, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/di/cdiyheupjrp7yxy33u2a4o64u6qgqab7q4puzpnuoektobali3tm.py
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
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (163072*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qf/cqfipppewwqfzabfiexsbo7vxpwjjqrh2em6767jyde5bxa2j4yt.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10816
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (163072*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (832*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/cx/ccxrwsp63ifwfxijgsvpy4lq2wmcblx2ymsfjq6c735wqresvii7.py
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
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 832
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


# kernel path: /tmp/torchinductor_youkaichao/cv/ccvcplrkiqhavy6hvsomms5xdfwtiudzre5geac6qwiofmpsnmdz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 832
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (163072*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (832*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (832*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gz/cgz235ggvv4i6euhluypgzthhwy3ugm2w73dmrnhecz6tste5idz.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
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
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ui/cuiywhycrtqruo3du2bqkxp4rx4cv4c53u3pdeuobhp4clnqtlxa.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13312
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
        tmp3 = tl.load(in_ptr0 + (x1 + (1024*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((196*x1) + (200704*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + ((196*x1) + (200704*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.where(tmp5, tmp4, tmp8)
        tmp10 = tl.load(in_ptr3 + (x1 + (1024*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/il/ciln4wdqanxpiciz57rqv5c7c2njvgx6yviljdthc6mlajomjhjb.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
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


# kernel path: /tmp/torchinductor_youkaichao/uv/cuvum4khqkkvxaqwahyoavmy46annfxzon2m3ioz5srabhttrl7m.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (196*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (196*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ij/cij3tgnm3suiqbkcerc6opzjh5qzq3mkwquui26lu3ymrg23ugdo.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1352
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
        tmp3 = tl.load(in_ptr0 + (x1 + (104*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (40768 + (196*x1) + (81536*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/om/coms52wbn6vtji5aucfl37amtnmijofb2eaviajq6yxrcd5kwdsr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 104
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


# kernel path: /tmp/torchinductor_youkaichao/of/cofjbbonndhfoabviazvg2r7apiy7gxtiq2irfm64ov5zo2ekcne.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1352
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 104)
    x0 = xindex % 104
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (104*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (40768 + (196*x0) + (81536*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.load(in_ptr2 + (x0 + (104*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/rx/crxnj6jsnz4tfmlogfgbe7vl5vhybnski6b26o2gwxw6wvigs22m.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 104
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (104*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cf/ccfxji6hcp4bqfz6yqeya5fzihzcershbv5evygultc2wxfrwxv3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 104
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
    tmp0 = tl.load(in_ptr0 + (x2 + (104*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (40768 + y0 + (196*x2) + (81536*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (104*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (104*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dq/cdqman6ogyjm72rxdq4nehaa47jawsyi5ybn7dr2etuhojr36nlt.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 104
    rnumel = 1568
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
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (104*r3)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (20384 + r1 + (196*x0) + (81536*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1 + (196*x0) + (20384*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2h/c2huhoutsevc2rxctxkj42pzyv5dwh7csoxygblpf7knsey2hbjr.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1352
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (104*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (20384 + (196*x1) + (81536*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + ((196*x1) + (20384*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = 0.0
        tmp8 = tl.where(tmp3, tmp7, tmp6)
        tmp9 = tl.load(in_ptr3 + (x1 + (104*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/uf/cufv7flkgyzswyzote373s5udj2ugnarb2kkpqpnlp2icnyw7hva.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 104
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


# kernel path: /tmp/torchinductor_youkaichao/he/chesniqebgma62itr74y2h4piwms22n5nfkx2qddfok2w36i47hk.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 104
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
    tmp0 = tl.load(in_ptr0 + (x2 + (104*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (20384 + y0 + (196*x2) + (81536*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (196*x2) + (20384*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (104*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.0006377551020408163
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr0 + (x2 + (104*y3)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yg/cyg2ziu54a4jktil7tguzdncmsciyvx4bmiwfn24hp74zlbxwcfu.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 104
    rnumel = 1568
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
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (104*r3)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (81536*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1 + (196*x0) + (20384*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qq/cqqehaa2buafbchi4w2q5gvi6ulq5knnyhtrrdmw7ipgzk354eqb.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1352
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (104*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((196*x1) + (81536*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + ((196*x1) + (20384*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = 0.0
        tmp8 = tl.where(tmp3, tmp7, tmp6)
        tmp9 = tl.load(in_ptr3 + (x1 + (104*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/p3/cp3fciyn72l2t4fncbfbhpk2i4mwym7zfmowofhu5nq5tjjxcv22.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 104
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
    tmp0 = tl.load(in_ptr0 + (x2 + (104*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (196*x2) + (81536*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (196*x2) + (20384*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (104*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.0006377551020408163
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr0 + (x2 + (104*y3)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/64/c644uutmq6ey2k522usweidkgfs4q4fopx3nv6ilnm463eug25a7.py
# Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]

triton_poi_fused_cat_threshold_backward_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_threshold_backward_51', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3328
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 416
    y1 = (yindex // 416)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (416*x2) + (81536*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 104, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (196*y0) + (20384*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 208, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-20384) + x2 + (196*y0) + (20384*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1, 1], 312, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + ((-40768) + x2 + (196*y0) + (20384*y1)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp1 >= tmp17
    tmp24 = tl.full([1, 1], 416, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.load(in_out_ptr0 + (x2 + (196*y3)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp12, tmp15, tmp29)
    tmp31 = tl.where(tmp5, tmp8, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp0, tmp32, tmp31)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp33, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4t/c4tub7l5s5vl6ed527d22k5sj5xbs2r6xcyek4fslncuozwwnafy.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 416
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (81536*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ma/cmaqovpru6ag72griiohh3pgcxvqpd64ul7t5omkvds4qlzfoz3d.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5408
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (81536*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (416*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/a2/ca2l5jytx2fosaslk2z3b5akfoq6t4lx7h66zcsdlya2u7tctkzq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 416
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


# kernel path: /tmp/torchinductor_youkaichao/qi/cqii4w3ijg3ek6zeu6aharrbt4uoqttgzlok4szmezavgvgzppin.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 416
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (81536*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (416*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (416*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nz/cnzr5gerd62g422vpp4rh44duj6otkvjs5jlyvgcmicaggtp4k5o.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_56', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (1024*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jt/cjtkwlrawfgvvidfkhonwesgtqjue7llfksish7bioxwedfalfeo.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zh/czhcd6c7q5lz4pjap4qz4swmv4roapk52itenbdrypvl24agekec.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13312
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (200704*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (1024*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/gj/cgjo7wgorafxk75bzlqwdzcwv57pvw7hxba2g6hfhmsfxktxcbvc.py
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
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ez/cezsjz3ngxo4szyta2j4aok6ljuj7v3rekq2olmvtb6ndobn4stx.py
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
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_60', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (1024*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5v/c5vdn4vomuwcmnwbxf5rc3sm7kxrfi5znukdwiv2phbcp3wha5jf.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_61', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (1024*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3o/c3o6j3zy45t3xlsez5olz7fendjbtavdgnu2mv24ab3c5efq6jxq.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13312
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (1024*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((196*x1) + (200704*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + ((196*x1) + (200704*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.where(tmp5, tmp4, tmp8)
        tmp10 = tl.load(in_ptr3 + (x1 + (1024*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr4 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp9 * tmp12
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp19 = tl.load(in_ptr5 + (x1 + (1024*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr6 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp9 * tmp21
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


# kernel path: /tmp/torchinductor_youkaichao/hv/chvw2k5hyuah4ctalmfsiuzcqtqad757egmdtwvtuaq6bezdyv7h.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_63 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16, 17))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (196*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (196*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr12 + (x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
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
    tmp26 = tmp24 - tmp25
    tmp28 = tmp27 * tmp11
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 * tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = tmp6 - tmp32
    tmp34 = tmp33 - tmp19
    tmp36 = tmp29 * tmp35
    tmp37 = tmp34 * tmp36
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp23, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (1024*y3)), tmp37, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3r/c3roof62xa5o4ysb7jhbwj7zfpoc52tzlslgtetbfkh5otcch7hb.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 652288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 28
    x1 = (xindex // 28) % 28
    x2 = (xindex // 784) % 104
    x3 = (xindex // 81536)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (61152 + (14*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2)))))) + (14*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2))))) >= 0, 0, 14))) + (196*x2) + (81536*x3) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) >= 0, 0, 14))), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (61152 + (14*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2)))))) + (14*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2))))) >= 0, 0, 14))) + (196*x2) + (81536*x3) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) >= 0, 0, 14))), xmask)
    tmp18 = tl.load(in_ptr0 + (61152 + (14*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2)))))) + (14*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2))))) >= 0, 0, 14))) + (196*x2) + (81536*x3) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) >= 0, 0, 14))), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (61152 + (14*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2)))))) + (14*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x1) // 2))))) >= 0, 0, 14))) + (196*x2) + (81536*x3) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(14, 1 + ((1 + x0) // 2))))) >= 0, 0, 14))), xmask)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (x1 // 2))
    tmp3 = tl.math.min(14, 1 + ((1 + x1) // 2))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (x0 // 2))
    tmp6 = tl.math.min(14, 1 + ((1 + x0) // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.math.max(0, (x0 // 2)))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 1 + (tl.math.max(0, (x1 // 2)))
    tmp21 = tmp20 < tmp3
    tmp22 = tmp21 & tmp7
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = tmp21 & tmp14
    tmp28 = tmp24 + tmp26
    tmp29 = tl.where(tmp27, tmp28, tmp24)
    tl.store(out_ptr0 + (x6), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j5/cj5zrft5ygijvu3qbj4qiotdgyqc5e3njlmatgdm3qrueucdppf6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1352
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
        tmp3 = tl.load(in_ptr0 + (x1 + (104*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (20384 + (196*x1) + (81536*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/f5/cf5pi7tdfkryos63ojnvrptblmfwnh7b7llgibinjvk6crh5io3m.py
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1352
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 104)
    x0 = xindex % 104
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (104*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + (20384 + (196*x0) + (81536*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.load(in_ptr2 + (x0 + (104*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/v3/cv3qtcnzsktsmdgdedquzvgrwm76n662ydzowcm3ckyp2t2fns2y.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 104
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
    tmp0 = tl.load(in_ptr0 + (x2 + (104*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (20384 + y0 + (196*x2) + (81536*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (104*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (104*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oi/coitbdnpqxws2hmmvwlonth3kjpi7vyikarolitdvfdwtuepmnvu.py
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1352
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
        tmp3 = tl.load(in_ptr0 + (x1 + (104*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((196*x1) + (81536*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/np/cnps46bijkgzthwjwtawme3aiqcss267nycr6pmwsq7xyukw63dd.py
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1352
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 104)
    x0 = xindex % 104
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (104*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp4 = tl.load(in_ptr1 + ((196*x0) + (81536*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp7 = tl.load(in_ptr2 + (x0 + (104*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/yh/cyhklnsdphpcz73oqvyyjjy2a2iurjdcq2hfqw7ke7zvf26i5lbo.py
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
    size_hints=[2048, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 104
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
    tmp0 = tl.load(in_ptr0 + (x2 + (104*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (196*x2) + (81536*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (104*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (104*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2h/c2h737vuxnfdoibo3ovqjfm63fffkhqtyao2vb7j3o2m4fznw3hf.py
# Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]

triton_poi_fused_cat_threshold_backward_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_threshold_backward_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3328
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 416
    y1 = (yindex // 416)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (416*x2) + (326144*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 104, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (784*y0) + (81536*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 208, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-81536) + x2 + (784*y0) + (81536*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1, 1], 312, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + ((-163072) + x2 + (784*y0) + (81536*y1)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp1 >= tmp17
    tmp24 = tl.full([1, 1], 416, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.load(in_ptr4 + ((-244608) + x2 + (784*y0) + (81536*y1)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp12, tmp15, tmp29)
    tmp31 = tl.where(tmp5, tmp8, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp0, tmp32, tmp31)
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp33, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qc/cqcrudr2qn6x4vkmq7exh6i45jj4fsy7ivnbebwv4b2eexm4qaew.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 416
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (326144*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w5/cw5m23mlntjcxpkgch55lws766it4lggeb77t5wkgquzxd6paveb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 20384
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
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (326144*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (416*r2) + (53248*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2l/c2lnvegsqt6flgdv7rta7ica3k7gxdjb6r5o5xow7di56eubp7np.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 416
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


# kernel path: /tmp/torchinductor_youkaichao/52/c527embxc7jqyybooieud6qbd6sx7wsfjyrus2j6fz73g6osjqdd.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 416
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (416*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (416*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jo/cjop4i5oilhqsw7oy36xibilm37odzom7as6vsdqw5juirwfeaqq.py
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
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/rx/crxvjwxe3nb4mb33ikbw556htvlrle7bnoogezgv6kxrsgjg4xy7.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
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
        tmp0 = tl.load(in_ptr0 + (x1 + (512*r2) + (65536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((784*x1) + (401408*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((784*x1) + (401408*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (512*r2) + (65536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ar/carptrvoejh73adq3u42tyfnu6dqenxmwhbugccaxsqkhcejkkkd.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_78 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_78', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wx/cwx7hij6cgej6kgcfxde3qh5nepoqwjfm3qx6rqpdalwnrasbgpp.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_79 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (784*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/io/ciotjnuxzbnfqurgqn73vzugxmc3ulfbhasda5ejihjs3iswdvxz.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2548
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
        tmp0 = tl.load(in_ptr0 + (x1 + (52*r2) + (6656*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (81536 + (784*x1) + (163072*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ur/cur4nmfppn6idvha3ufk53f5xouvvbnz6ttbxigdc72s5jvjf6gf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 52
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


# kernel path: /tmp/torchinductor_youkaichao/dg/cdgud7fsq73v3xxahw65ynnz5xc42ycidmian6kszjtode2nz6ba.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_82', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2548
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 52
    x1 = (xindex // 52)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (52*r2) + (6656*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (81536 + (784*x0) + (163072*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (52*r2) + (6656*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/mw/cmw6acgjz5nbtv3uw655gyinkaf7fpg5zmbsawb4wve6w3e3kdvm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_83', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 52
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (52*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/22/c22a3jypgrmbx5bbkn4ou63yugqbkgfiqvn67c2w25qpn7qxo2fo.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 52
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
    tmp0 = tl.load(in_ptr0 + (x2 + (52*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (81536 + y0 + (784*x2) + (163072*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (52*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (52*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rj/crjwsk7qhmleqldthdtpzyw532tuxvzjpzfh7bgmxe3ls2bgwxiu.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_85 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 52
    rnumel = 6272
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
        tmp0 = tl.load(in_ptr0 + (x0 + (52*r3)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (40768 + r1 + (784*x0) + (163072*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1 + (784*x0) + (40768*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h2/ch2c7qun3rwirn6mts3ynwwjtcfwgtx6jdaw4zhyhtkb3wicmkum.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_86 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_86', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2548
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (52*r2) + (6656*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (40768 + (784*x1) + (163072*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + ((784*x1) + (40768*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x1 + (52*r2) + (6656*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/xj/cxje2lrvx35xbtpymlzuii2uui4xrdrbw4yvcdt4bqgq7qosyc2n.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 52
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


# kernel path: /tmp/torchinductor_youkaichao/rn/crnrermmtumlcqqyqlb64atcqzv35oruuhwravtcdbj3wgpre2zr.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_88 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 52
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
    tmp0 = tl.load(in_ptr0 + (x2 + (52*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (40768 + y0 + (784*x2) + (163072*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (784*x2) + (40768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (52*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.00015943877551020407
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr0 + (x2 + (52*y3)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6d/c6dsueyljq2uq4swxgc7epu55lpbofpsgdghmqcxotyv7nmkc52c.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_89 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 52
    rnumel = 6272
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
        tmp0 = tl.load(in_ptr0 + (x0 + (52*r3)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r1 + (784*x0) + (163072*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1 + (784*x0) + (40768*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oy/coyzqmxxif6ni3bfeykrdxstaurisku5bn223fqkgtmaxeru6v37.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_90 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2548
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (52*r2) + (6656*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((784*x1) + (163072*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + ((784*x1) + (40768*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x1 + (52*r2) + (6656*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/cb/ccbcmbiyq4sk7a2j7pzaxbvtc3kzdbefhxwwcu7x3ntqzp5wcgvj.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_91 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_91', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 52
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
    tmp0 = tl.load(in_ptr0 + (x2 + (52*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (784*x2) + (163072*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (784*x2) + (40768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (52*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.00015943877551020407
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr0 + (x2 + (52*y3)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jc/cjcco7rlc3tiueldjv35dtqcpup5rs7wqgaoiidxlvblliozxqev.py
# Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]

triton_poi_fused_cat_threshold_backward_92 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_threshold_backward_92', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (208*x2) + (163072*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 52, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (784*y0) + (40768*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 104, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-40768) + x2 + (784*y0) + (40768*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1, 1], 156, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + ((-81536) + x2 + (784*y0) + (40768*y1)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp1 >= tmp17
    tmp24 = tl.full([1, 1], 208, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.load(in_out_ptr0 + (x2 + (784*y3)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp12, tmp15, tmp29)
    tmp31 = tl.where(tmp5, tmp8, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp0, tmp32, tmp31)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp33, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ky/ckyiwzgj67ilfk2erbxypt2zjpgd6l7ofrpyah5cu4jtyf2l3pxb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_93', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 208
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (163072*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6d/c6dq7qvvhwpux6jah3cb2iqg24lpq5hizvvmtdmrs5xrbtxbhbvl.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_94', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10192
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
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (163072*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (208*r2) + (26624*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pn/cpn63sbv3b3iffc3llioxeyx3pumsgx5bpvf4yus7qmz3kpytngp.py
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
    size_hints=[256, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 208
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


# kernel path: /tmp/torchinductor_youkaichao/ki/ckiroheefccexfqcrmkdtyny7rtnmys47e527rnnp7hityx4dpo2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_96 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_96', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 208
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (163072*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (208*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (208*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wp/cwpsyx3mrslbwlpuquqlqew2ylormiez3wysjjpt4jqoxlc3ny6h.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_97 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_97', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (512*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4v/c4vsom3q4hf3mhnozzrjqwt4ydwksferjevbazeuz5huqg4sd44l.py
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
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_98', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bi/cbixf5v4qqvaogosc54seplg2n3qn2jwnifb7k37uz3foagzdpaw.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_99', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
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
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (401408*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (512*r2) + (65536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qa/cqalxkmgvwpphh352zv4ht47otxnd3zxa3yfxlochfymacp4oblu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_100 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_100', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cb/ccbzvt4m7eh4f7hwigyydzliqbjovfn57kkl6dkucpj2iwtwnfjc.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_101', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
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
    tmp9 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (401408*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (512*r2) + (65536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (512*r2) + (65536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4v/c4veobo2kaseqj6ujxdezgzjgjfy4prtc425wxmt27a4vqrm7zua.py
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
    size_hints=[8192, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_102', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
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
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (512*y3)), tmp31, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q3/cq3jw7qms7wlsvbqaqfnmpbd66hpgaenydwwar25eupfjbjjwmv6.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_103 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_103', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1304576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 56
    x1 = (xindex // 56) % 56
    x2 = (xindex // 3136) % 52
    x3 = (xindex // 163072)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (122304 + (28*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2))))) >= 0, 0, 28))) + (784*x2) + (163072*x3) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (122304 + (28*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2)))))) + (28*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2))))) >= 0, 0, 28))) + (784*x2) + (163072*x3) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) >= 0, 0, 28))), None)
    tmp18 = tl.load(in_ptr0 + (122304 + (28*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2)))))) + (28*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2))))) >= 0, 0, 28))) + (784*x2) + (163072*x3) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) >= 0, 0, 28))), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (122304 + (28*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2)))))) + (28*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x1) // 2))))) >= 0, 0, 28))) + (784*x2) + (163072*x3) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(28, 1 + ((1 + x0) // 2))))) >= 0, 0, 28))), None)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (x1 // 2))
    tmp3 = tl.math.min(28, 1 + ((1 + x1) // 2))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (x0 // 2))
    tmp6 = tl.math.min(28, 1 + ((1 + x0) // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.math.max(0, (x0 // 2)))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 1 + (tl.math.max(0, (x1 // 2)))
    tmp21 = tmp20 < tmp3
    tmp22 = tmp21 & tmp7
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = tmp21 & tmp14
    tmp28 = tmp24 + tmp26
    tmp29 = tl.where(tmp27, tmp28, tmp24)
    tl.store(out_ptr0 + (x6), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7v/c7vtqzjiwqdhhspf5dhykyl34xmblxnka66w7p4emz5gr3dj26dv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_104 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_104', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2548
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
        tmp0 = tl.load(in_ptr0 + (x1 + (52*r2) + (6656*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (40768 + (784*x1) + (163072*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kf/ckf7w6vss2q5plmitkficatz5sty76swlrd7rcbhgnntgexcugj6.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_105', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2548
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 52
    x1 = (xindex // 52)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (52*r2) + (6656*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (40768 + (784*x0) + (163072*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (52*r2) + (6656*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/nu/cnuomvz6z62kpdyq2uiuge6esyzeaalojdyr6evjcv5iwarh76xc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_106 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_106', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 52
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
    tmp0 = tl.load(in_ptr0 + (x2 + (52*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (40768 + y0 + (784*x2) + (163072*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (52*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (52*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/el/celh7ja3obflrgqurcwfufreq4q5xlzq2rr2zhrwt3b52amwkoqm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_107 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_107', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2548
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
        tmp0 = tl.load(in_ptr0 + (x1 + (52*r2) + (6656*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((784*x1) + (163072*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oa/coaoxj7lvw36l6kndpozst7cpn277ejgds3eoegrwrxhr5xz73aq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_108 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_108', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2548
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 52
    x1 = (xindex // 52)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (52*r2) + (6656*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((784*x0) + (163072*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (52*r2) + (6656*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/22/c22c4kvc4aslltc7lmw5o3ohg3dtfhg5yutgley6lrtc5krzgdjk.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_109 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_109', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 52
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
    tmp0 = tl.load(in_ptr0 + (x2 + (52*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (784*x2) + (163072*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (52*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (52*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u4/cu4ihuckkdi6m4crmm6xrjpds46no5qo22k4gwpaqaq424foqxnj.py
# Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]

triton_poi_fused_cat_threshold_backward_110 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_threshold_backward_110', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1664
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (208*x2) + (652288*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 52, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (3136*y0) + (163072*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 104, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-163072) + x2 + (3136*y0) + (163072*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1, 1], 156, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + ((-326144) + x2 + (3136*y0) + (163072*y1)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp1 >= tmp17
    tmp24 = tl.full([1, 1], 208, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.load(in_ptr4 + ((-489216) + x2 + (3136*y0) + (163072*y1)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp12, tmp15, tmp29)
    tmp31 = tl.where(tmp5, tmp8, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp0, tmp32, tmp31)
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp33, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3c/c3cjoaqo24525y4oymnbzsj2ory7wq6jov353lcz2vbekmccrbp6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_111 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_111', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 208
    x1 = (xindex // 208)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (652288*(r2 // 3136)) + (1304576*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7h/c7h67k3okbjqpvneiyoqpjdscfhc3qt2cjxt4uyfo4pofguxabcg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_112 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_112', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 208
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (208*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jf/cjf7xpdulekiakogf55shoktm7m42nsdplw2zfzkwqmjpec4trgp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_113 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_113', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 40768
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
        tmp0 = tl.load(in_ptr0 + ((3136*x1) + (652288*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (208*r2) + (26624*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/po/cporuhkcvw3lci2ke6vx6pcqktsfzhsazxziudlcib3ftca25wcl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_114 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_114', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 208
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


# kernel path: /tmp/torchinductor_youkaichao/5h/c5hdu547b6wyltjafqecwxneddr4mcp2yamlg37dc7gdpdf7u4fh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_115 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_115', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 208
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (652288*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (208*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (208*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4u/c4ujjm3ieplt6cmtr6dhsmf2q7mnv65rn4k5abr6z62yca2ncofa.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_116 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_116', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 25088
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
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/g7/cg7jrc6d4d6m2xw5fcx4w7t53o7x337t4qaokeakjaxf4z3tdmg4.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_117 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_117', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
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
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x1) + (802816*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x1) + (802816*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/c6/cc6pnuzcyo3xxkc5ef4vrag55tu3a5bqjkpz7j2z3ttcorajj5ll.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_118 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_118', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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


# kernel path: /tmp/torchinductor_youkaichao/dy/cdypkludl5z2ggc6zxoyyp47lprhq4bdhqtxelsr3gkm7nryzc2p.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_119 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_119', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (802816*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (3136*x2) + (802816*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yl/cylyssadchvskppsi7muc2t7q2wp3ejwn3l53crtvouzqydqzxti.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_120 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_120', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5096
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
        tmp0 = tl.load(in_ptr0 + (x1 + (26*r2) + (3328*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (163072 + (3136*x1) + (326144*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ho/chofmzg3jyt6lsz64sn5fsddnals6exjogywly4rn5huiy6k4wwb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_121 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_121', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 26
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


# kernel path: /tmp/torchinductor_youkaichao/7b/c7baglzahihq5gfuqdyrmtnrxneghxj4ykaog2rrgf6potucm5kq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_122 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_122', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 26
    x1 = (xindex // 26)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (26*r2) + (3328*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (163072 + (3136*x0) + (326144*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (26*r2) + (3328*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/p2/cp2xzog6ctp5opmurru4hzw3bwj3pcgzywmdfkgj2x3pzlmoombu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_123 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_123', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 26
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
        tmp0 = tl.load(in_ptr0 + (x0 + (26*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u5/cu56jbk52ttq3tinv3ssngshppa5bl6mmnlsjlji6mob7yhhwkv3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_124 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_124', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 26
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
    tmp0 = tl.load(in_ptr0 + (x2 + (26*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (163072 + y0 + (3136*x2) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (26*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (26*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yp/cyps32ann7dtct7cpv23ztkwqanf4ysrdkf4ejvrk7nfgtyrfpcr.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_125 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_125', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 104
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 26
    x1 = (xindex // 26)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (26*r2) + (163072*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (81536 + (3136*x0) + (326144*(r2 // 3136)) + (652288*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + ((3136*x0) + (81536*(r2 // 3136)) + (163072*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/he/chelypxhauqfbvphud6vswczksaeamwtzf246f2ggu57hkup2gyi.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_126 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_126', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 26
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (26*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wy/cwytthcwf4l7buyn7mncq3gfenzfh5mr6tjx24avull4e52h2bkf.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_127 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_127', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (26*r2) + (3328*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (81536 + (3136*x1) + (326144*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + ((3136*x1) + (81536*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x1 + (26*r2) + (3328*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bz/cbzxckrxgdgvuvd65ghigft7ijdezj633uics4b4wxuirnaa2kzp.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_128 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_128', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 26
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


# kernel path: /tmp/torchinductor_youkaichao/gf/cgfyeulrftmsytwumcwbmupa5axgfvwgg7ilv2gaxejm4r7uh7ly.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_129 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_129', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 26
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
    tmp0 = tl.load(in_ptr0 + (x2 + (26*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (81536 + y0 + (3136*x2) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (3136*x2) + (81536*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (26*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 3.985969387755102e-05
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr0 + (x2 + (26*y3)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5t/c5tsbbimdu6s73t7xc6riceg6b25dwjhcc7julhsofrs6nni6ils.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_130 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_130', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 104
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 26
    x1 = (xindex // 26)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (26*r2) + (163072*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((3136*x0) + (326144*(r2 // 3136)) + (652288*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + ((3136*x0) + (81536*(r2 // 3136)) + (163072*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p6/cp6mrprjibgltefubrarbvsohgnures7da5kuipjxh2xctbwtmj3.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_131 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_131', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (26*r2) + (3328*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((3136*x1) + (326144*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + ((3136*x1) + (81536*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x1 + (26*r2) + (3328*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/vh/cvhnwau4dkr7t7x5kapdm5wxkew5wnuqi7r4toy6z5l6umv7jtoj.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_132 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_132', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 26
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
    tmp0 = tl.load(in_ptr0 + (x2 + (26*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (3136*x2) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0 + (3136*x2) + (81536*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (26*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 3.985969387755102e-05
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr0 + (x2 + (26*y3)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s6/cs6wlg2e2njhhgjim5qk4kwreo7yl3itel36rwaoe5mfxcmh7n5x.py
# Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]

triton_poi_fused_cat_threshold_backward_133 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_threshold_backward_133', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (104*x2) + (326144*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 26, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (3136*y0) + (81536*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 52, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-81536) + x2 + (3136*y0) + (81536*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1, 1], 78, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + ((-163072) + x2 + (3136*y0) + (81536*y1)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp1 >= tmp17
    tmp24 = tl.full([1, 1], 104, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp12, tmp15, tmp29)
    tmp31 = tl.where(tmp5, tmp8, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp0, tmp32, tmp31)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (3136*y3)), tmp33, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/25/c254dzlpy32bwjg7ukwrezw3w5rsho6ijlhrzkj4svnjvb3exqkh.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_134 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_134', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 416
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 104
    x1 = (xindex // 104)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (326144*(r2 // 3136)) + (652288*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/up/cupwuhroyam3hklu7lgh2olv4i3smah6adntkyicbhhpgjohs2hl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_135 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_135', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 104
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (104*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gi/cgibzd3p4gmspylpb4bw2bet4ybjqhbw4inicfqny35gicdh3kry.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_136 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_136', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 20384
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
        tmp0 = tl.load(in_ptr0 + ((3136*x1) + (326144*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (104*r2) + (13312*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2s/c2sas42roim5h3uyzwanur5wiem45lxrvq26skks3rrrqyif6rth.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_137 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_137', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 104
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


# kernel path: /tmp/torchinductor_youkaichao/35/c35wm52zujst5w6ce2l3vw7mzetr67s7hs4jql3ldzqxpm67dsi6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_138 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_138', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 104
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (104*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (104*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hj/chjxq6xspjr3lksyk3bcpks7d7oau3yyxlhg5zsphwzyvbiz6m5b.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_139 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_139', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (802816*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (256*x2) + (802816*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (3136*y3)), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/77/c777l5bj4cjyqztkqucbfkawo2gks52iwz2ea24ymmtmkzsy675c.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_140 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_140', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uj/cujgh62fd22ekm53deimduy72rf2dngrg2ae5pk6wkn47z2ix7pl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_141 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_141', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
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
        tmp0 = tl.load(in_ptr0 + ((3136*x1) + (802816*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hd/chdwmc5u4edozoyymmkuucbo2hhqi4vn3gqtpidp7sultv2tjb5m.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_142 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_142', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (802816*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kl/cklnuo34j2m2uglfiwenad5wefhmitoas55k7zg5y3nu5i6qzk7c.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_143 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_143', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
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
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x1) + (802816*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x1) + (802816*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr5 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp6 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tq/ctqvdsayl4h6iklzaqonxxnusy625ukijythz5igotzuvoey2hfx.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_144 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16, 17))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_144', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (802816*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (3136*x2) + (802816*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr12 + (x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
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
    tmp26 = tmp24 - tmp25
    tmp28 = tmp27 * tmp11
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 * tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = tmp6 - tmp32
    tmp34 = tmp33 - tmp19
    tmp36 = tmp29 * tmp35
    tmp37 = tmp34 * tmp36
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp23, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (256*y3)), tmp37, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v6/cv65tnzdehr2h6oyicvgudfpvw2u4ic6ne5xuvz4ekw2gxaveues.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]

triton_poi_fused_avg_pool2d_backward_145 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_145', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 652288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 56
    x1 = (xindex // 56) % 56
    x2 = (xindex // 3136) % 26
    x3 = (xindex // 81536)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (244608 + (56*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(56, 2 + x1))))) + (3136*x2) + (326144*x3) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(56, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (244608 + (56*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(56, 2 + x1))))) + (3136*x2) + (326144*x3) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(56, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (244608 + (56*(tl.math.min(tl.math.max(0, (-1) + x1), (-1) + (tl.math.min(56, 2 + x1))))) + (3136*x2) + (326144*x3) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(56, 2 + x0))))), xmask)
    tmp25 = tl.load(in_ptr0 + (244608 + (56*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(56, 2 + x1))))) + (3136*x2) + (326144*x3) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(56, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (244608 + (56*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(56, 2 + x1))))) + (3136*x2) + (326144*x3) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(56, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (244608 + (56*(tl.math.min(1 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(56, 2 + x1))))) + (3136*x2) + (326144*x3) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(56, 2 + x0))))), xmask)
    tmp42 = tl.load(in_ptr0 + (244608 + (56*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(56, 2 + x1))))) + (3136*x2) + (326144*x3) + (tl.math.min(tl.math.max(0, (-1) + x0), (-1) + (tl.math.min(56, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + (244608 + (56*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(56, 2 + x1))))) + (3136*x2) + (326144*x3) + (tl.math.min(1 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(56, 2 + x0))))), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + (244608 + (56*(tl.math.min(2 + (tl.math.max(0, (-1) + x1)), (-1) + (tl.math.min(56, 2 + x1))))) + (3136*x2) + (326144*x3) + (tl.math.min(2 + (tl.math.max(0, (-1) + x0)), (-1) + (tl.math.min(56, 2 + x0))))), xmask)
    tmp1 = tmp0 / 9
    tmp2 = tl.math.max(0, (-1) + x1)
    tmp3 = tl.math.min(56, 2 + x1)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (-1) + x0)
    tmp6 = tl.math.min(56, 2 + x0)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp11 / 9
    tmp13 = 1 + (tl.math.max(0, (-1) + x0))
    tmp14 = tmp13 < tmp6
    tmp15 = tmp4 & tmp14
    tmp16 = tmp10 + tmp12
    tmp17 = tl.where(tmp15, tmp16, tmp10)
    tmp19 = tmp18 / 9
    tmp20 = 2 + (tl.math.max(0, (-1) + x0))
    tmp21 = tmp20 < tmp6
    tmp22 = tmp4 & tmp21
    tmp23 = tmp17 + tmp19
    tmp24 = tl.where(tmp22, tmp23, tmp17)
    tmp26 = tmp25 / 9
    tmp27 = 1 + (tl.math.max(0, (-1) + x1))
    tmp28 = tmp27 < tmp3
    tmp29 = tmp28 & tmp7
    tmp30 = tmp24 + tmp26
    tmp31 = tl.where(tmp29, tmp30, tmp24)
    tmp33 = tmp32 / 9
    tmp34 = tmp28 & tmp14
    tmp35 = tmp31 + tmp33
    tmp36 = tl.where(tmp34, tmp35, tmp31)
    tmp38 = tmp37 / 9
    tmp39 = tmp28 & tmp21
    tmp40 = tmp36 + tmp38
    tmp41 = tl.where(tmp39, tmp40, tmp36)
    tmp43 = tmp42 / 9
    tmp44 = 2 + (tl.math.max(0, (-1) + x1))
    tmp45 = tmp44 < tmp3
    tmp46 = tmp45 & tmp7
    tmp47 = tmp41 + tmp43
    tmp48 = tl.where(tmp46, tmp47, tmp41)
    tmp50 = tmp49 / 9
    tmp51 = tmp45 & tmp14
    tmp52 = tmp48 + tmp50
    tmp53 = tl.where(tmp51, tmp52, tmp48)
    tmp55 = tmp54 / 9
    tmp56 = tmp45 & tmp21
    tmp57 = tmp53 + tmp55
    tmp58 = tl.where(tmp56, tmp57, tmp53)
    tl.store(out_ptr0 + (x6), tmp58, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bb/cbbo3uqydt77jx7rawx6nxubhcpmtrdgtyrj462muh7lhfcve2h6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_146 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_146', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5096
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
        tmp0 = tl.load(in_ptr0 + (x1 + (26*r2) + (3328*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (81536 + (3136*x1) + (326144*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xa/cxayf3vparn4mlxn52b2f75rcas4cps44rfwh67nhbilswb6p6xj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_147 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_147', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 26
    x1 = (xindex // 26)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (26*r2) + (3328*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (81536 + (3136*x0) + (326144*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (26*r2) + (3328*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/gb/cgbpi3nrlqiq4aviykwelbghnmsf6cawi7ndcbfb55qromo5qjnl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_148 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_148', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 26
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
    tmp0 = tl.load(in_ptr0 + (x2 + (26*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (81536 + y0 + (3136*x2) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (26*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (26*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nm/cnmgnjuyg7l5v6hnwev5riifvfzas7ritmwlhr2oa223huk357l3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_149 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_149', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5096
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
        tmp0 = tl.load(in_ptr0 + (x1 + (26*r2) + (3328*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((3136*x1) + (326144*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gl/cgldii7matmhjpeobhj3ecvgsno7g4eaulcwscitm7o26kxu2go7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_150 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_150', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 26
    x1 = (xindex // 26)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (26*r2) + (3328*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((3136*x0) + (326144*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (26*r2) + (3328*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/jk/cjkm6lunqapwvpourrkz5aidv7ednjnksgw3tebqagolqgbqjutv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_151 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_151', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 26
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
    tmp0 = tl.load(in_ptr0 + (x2 + (26*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (3136*x2) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (26*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (26*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ru/cruizgn4pymfw4c222k77di6i5fru4h66dgghxawsjuihc2tqdb6.py
# Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]

triton_poi_fused_cat_threshold_backward_152 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_threshold_backward_152', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (104*x2) + (326144*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 26, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (3136*y0) + (81536*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 52, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-81536) + x2 + (3136*y0) + (81536*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1, 1], 78, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + ((-163072) + x2 + (3136*y0) + (81536*y1)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp1 >= tmp17
    tmp24 = tl.full([1, 1], 104, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = tl.load(in_ptr4 + ((-244608) + x2 + (3136*y0) + (81536*y1)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp12, tmp15, tmp29)
    tmp31 = tl.where(tmp5, tmp8, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp0, tmp32, tmp31)
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp33, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/47/c47u2arsoks2ufhuyrqfgwvhzhpph76m76zrmgwwdvkmsyhmehmn.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_153 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_153', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oi/coi73pdndvckyafcgfjack2ibqbylmvpefvr6cbzmw5fh2mig67o.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_154 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_154', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/hk/chkjsh4rmxawuetrs77zkx36i43t2swh5ll4ztgqx7xzzfrx7j2n.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_155 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_155', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/uq/cuq2dhutrtuq6egwrmth4a6nzlvbqkdjl53cu7hza5u6uq6yfb2m.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_156 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_156', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/dy/cdyxb4kda32n5rkpj3677xnsxww7z757lthhyqtmxlox6ofb5eqa.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_157 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_157', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr1 + (x2), None)
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_281, primals_283, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_296, primals_298, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_311, primals_313, primals_314, primals_316, primals_317, primals_319, primals_320, primals_322, primals_323, primals_325, primals_326, primals_328, primals_329, primals_331, primals_332, primals_334, primals_335, primals_337, primals_338, primals_340, primals_341, primals_343, primals_344, primals_346, primals_347, primals_349, primals_350, primals_352, primals_353, primals_355, primals_356, primals_358, primals_359, primals_361, primals_362, primals_364, primals_365, primals_367, primals_368, primals_370, primals_371, primals_373, primals_374, primals_376, primals_377, primals_379, primals_380, primals_382, primals_383, primals_385, primals_386, primals_388, primals_389, primals_391, primals_392, primals_394, primals_395, primals_397, primals_398, primals_400, primals_401, primals_403, primals_404, primals_406, primals_407, primals_409, primals_410, primals_412, primals_413, primals_415, primals_416, primals_418, primals_419, primals_421, primals_422, primals_424, primals_425, primals_427, primals_428, primals_430, primals_431, primals_433, primals_434, primals_436, primals_437, primals_439, primals_440, primals_442, primals_443, primals_445, primals_446, primals_448, primals_449, primals_451, primals_452, primals_454, primals_455, primals_457, primals_458, primals_460, primals_461, primals_463, primals_464, primals_466, primals_467, primals_469, primals_470, primals_472, primals_473, primals_475, primals_476, primals_478, primals_479, primals_481, primals_482, primals_484, primals_485, primals_487, primals_488, primals_490, primals_491, primals_493, primals_494, primals_496, primals_497, primals_499, primals_500, primals_502, primals_503, primals_505, primals_506, primals_508, primals_509, primals_1023, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, getitem_10, convolution_2, squeeze_7, getitem_17, convolution_3, squeeze_10, getitem_24, convolution_4, squeeze_13, getitem_31, cat, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, getitem_42, convolution_8, squeeze_25, add_46, convolution_9, squeeze_28, add_52, convolution_10, squeeze_31, cat_1, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, getitem_72, convolution_13, squeeze_40, add_74, convolution_14, squeeze_43, add_80, convolution_15, squeeze_46, cat_2, convolution_16, squeeze_49, relu_15, convolution_17, squeeze_52, getitem_102, convolution_18, squeeze_55, getitem_109, convolution_19, squeeze_58, getitem_116, convolution_20, squeeze_61, getitem_123, cat_3, convolution_21, squeeze_64, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, getitem_134, convolution_24, squeeze_73, add_133, convolution_25, squeeze_76, add_139, convolution_26, squeeze_79, cat_4, convolution_27, squeeze_82, relu_25, convolution_28, squeeze_85, getitem_164, convolution_29, squeeze_88, add_161, convolution_30, squeeze_91, add_167, convolution_31, squeeze_94, cat_5, convolution_32, squeeze_97, relu_30, convolution_33, squeeze_100, getitem_194, convolution_34, squeeze_103, add_189, convolution_35, squeeze_106, add_195, convolution_36, squeeze_109, cat_6, convolution_37, squeeze_112, relu_35, convolution_38, squeeze_115, getitem_224, convolution_39, squeeze_118, getitem_231, convolution_40, squeeze_121, getitem_238, convolution_41, squeeze_124, getitem_245, cat_7, convolution_42, squeeze_127, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, getitem_256, convolution_45, squeeze_136, add_248, convolution_46, squeeze_139, add_254, convolution_47, squeeze_142, cat_8, convolution_48, squeeze_145, relu_45, convolution_49, squeeze_148, getitem_286, convolution_50, squeeze_151, add_276, convolution_51, squeeze_154, add_282, convolution_52, squeeze_157, cat_9, convolution_53, squeeze_160, relu_50, convolution_54, squeeze_163, getitem_316, convolution_55, squeeze_166, add_304, convolution_56, squeeze_169, add_310, convolution_57, squeeze_172, cat_10, convolution_58, squeeze_175, relu_55, convolution_59, squeeze_178, getitem_346, convolution_60, squeeze_181, add_332, convolution_61, squeeze_184, add_338, convolution_62, squeeze_187, cat_11, convolution_63, squeeze_190, relu_60, convolution_64, squeeze_193, getitem_376, convolution_65, squeeze_196, add_360, convolution_66, squeeze_199, add_366, convolution_67, squeeze_202, cat_12, convolution_68, squeeze_205, relu_65, convolution_69, squeeze_208, getitem_406, convolution_70, squeeze_211, add_388, convolution_71, squeeze_214, add_394, convolution_72, squeeze_217, cat_13, convolution_73, squeeze_220, relu_70, convolution_74, squeeze_223, getitem_436, convolution_75, squeeze_226, add_416, convolution_76, squeeze_229, add_422, convolution_77, squeeze_232, cat_14, convolution_78, squeeze_235, relu_75, convolution_79, squeeze_238, getitem_466, convolution_80, squeeze_241, add_444, convolution_81, squeeze_244, add_450, convolution_82, squeeze_247, cat_15, convolution_83, squeeze_250, relu_80, convolution_84, squeeze_253, getitem_496, convolution_85, squeeze_256, add_472, convolution_86, squeeze_259, add_478, convolution_87, squeeze_262, cat_16, convolution_88, squeeze_265, relu_85, convolution_89, squeeze_268, getitem_526, convolution_90, squeeze_271, add_500, convolution_91, squeeze_274, add_506, convolution_92, squeeze_277, cat_17, convolution_93, squeeze_280, relu_90, convolution_94, squeeze_283, getitem_556, convolution_95, squeeze_286, add_528, convolution_96, squeeze_289, add_534, convolution_97, squeeze_292, cat_18, convolution_98, squeeze_295, relu_95, convolution_99, squeeze_298, getitem_586, convolution_100, squeeze_301, add_556, convolution_101, squeeze_304, add_562, convolution_102, squeeze_307, cat_19, convolution_103, squeeze_310, relu_100, convolution_104, squeeze_313, getitem_616, convolution_105, squeeze_316, add_584, convolution_106, squeeze_319, add_590, convolution_107, squeeze_322, cat_20, convolution_108, squeeze_325, relu_105, convolution_109, squeeze_328, getitem_646, convolution_110, squeeze_331, add_612, convolution_111, squeeze_334, add_618, convolution_112, squeeze_337, cat_21, convolution_113, squeeze_340, relu_110, convolution_114, squeeze_343, getitem_676, convolution_115, squeeze_346, add_640, convolution_116, squeeze_349, add_646, convolution_117, squeeze_352, cat_22, convolution_118, squeeze_355, relu_115, convolution_119, squeeze_358, getitem_706, convolution_120, squeeze_361, add_668, convolution_121, squeeze_364, add_674, convolution_122, squeeze_367, cat_23, convolution_123, squeeze_370, relu_120, convolution_124, squeeze_373, getitem_736, convolution_125, squeeze_376, add_696, convolution_126, squeeze_379, add_702, convolution_127, squeeze_382, cat_24, convolution_128, squeeze_385, relu_125, convolution_129, squeeze_388, getitem_766, convolution_130, squeeze_391, add_724, convolution_131, squeeze_394, add_730, convolution_132, squeeze_397, cat_25, convolution_133, squeeze_400, relu_130, convolution_134, squeeze_403, getitem_796, convolution_135, squeeze_406, add_752, convolution_136, squeeze_409, add_758, convolution_137, squeeze_412, cat_26, convolution_138, squeeze_415, relu_135, convolution_139, squeeze_418, getitem_826, convolution_140, squeeze_421, add_780, convolution_141, squeeze_424, add_786, convolution_142, squeeze_427, cat_27, convolution_143, squeeze_430, relu_140, convolution_144, squeeze_433, getitem_856, convolution_145, squeeze_436, add_808, convolution_146, squeeze_439, add_814, convolution_147, squeeze_442, cat_28, convolution_148, squeeze_445, relu_145, convolution_149, squeeze_448, getitem_886, convolution_150, squeeze_451, add_836, convolution_151, squeeze_454, add_842, convolution_152, squeeze_457, cat_29, convolution_153, squeeze_460, relu_150, convolution_154, squeeze_463, getitem_916, convolution_155, squeeze_466, getitem_923, convolution_156, squeeze_469, getitem_930, convolution_157, squeeze_472, getitem_937, cat_30, convolution_158, squeeze_475, convolution_159, squeeze_478, relu_155, convolution_160, squeeze_481, getitem_948, convolution_161, squeeze_484, add_895, convolution_162, squeeze_487, add_901, convolution_163, squeeze_490, cat_31, convolution_164, squeeze_493, relu_160, convolution_165, squeeze_496, getitem_978, convolution_166, squeeze_499, add_923, convolution_167, squeeze_502, add_929, convolution_168, squeeze_505, cat_32, convolution_169, squeeze_508, view, permute_1, le, unsqueeze_682, le_1, unsqueeze_694, le_2, unsqueeze_706, le_3, unsqueeze_718, le_4, unsqueeze_730, unsqueeze_742, le_6, unsqueeze_754, le_7, unsqueeze_766, le_8, unsqueeze_778, le_9, unsqueeze_790, unsqueeze_802, unsqueeze_814, le_11, unsqueeze_826, le_12, unsqueeze_838, le_13, unsqueeze_850, le_14, unsqueeze_862, unsqueeze_874, le_16, unsqueeze_886, le_17, unsqueeze_898, le_18, unsqueeze_910, le_19, unsqueeze_922, unsqueeze_934, le_21, unsqueeze_946, le_22, unsqueeze_958, le_23, unsqueeze_970, le_24, unsqueeze_982, unsqueeze_994, le_26, unsqueeze_1006, le_27, unsqueeze_1018, le_28, unsqueeze_1030, le_29, unsqueeze_1042, unsqueeze_1054, le_31, unsqueeze_1066, le_32, unsqueeze_1078, le_33, unsqueeze_1090, le_34, unsqueeze_1102, unsqueeze_1114, le_36, unsqueeze_1126, le_37, unsqueeze_1138, le_38, unsqueeze_1150, le_39, unsqueeze_1162, unsqueeze_1174, le_41, unsqueeze_1186, le_42, unsqueeze_1198, le_43, unsqueeze_1210, le_44, unsqueeze_1222, unsqueeze_1234, le_46, unsqueeze_1246, le_47, unsqueeze_1258, le_48, unsqueeze_1270, le_49, unsqueeze_1282, unsqueeze_1294, le_51, unsqueeze_1306, le_52, unsqueeze_1318, le_53, unsqueeze_1330, le_54, unsqueeze_1342, unsqueeze_1354, le_56, unsqueeze_1366, le_57, unsqueeze_1378, le_58, unsqueeze_1390, le_59, unsqueeze_1402, unsqueeze_1414, le_61, unsqueeze_1426, le_62, unsqueeze_1438, le_63, unsqueeze_1450, le_64, unsqueeze_1462, unsqueeze_1474, le_66, unsqueeze_1486, le_67, unsqueeze_1498, le_68, unsqueeze_1510, le_69, unsqueeze_1522, unsqueeze_1534, le_71, unsqueeze_1546, le_72, unsqueeze_1558, le_73, unsqueeze_1570, le_74, unsqueeze_1582, unsqueeze_1594, le_76, unsqueeze_1606, le_77, unsqueeze_1618, le_78, unsqueeze_1630, le_79, unsqueeze_1642, unsqueeze_1654, le_81, unsqueeze_1666, le_82, unsqueeze_1678, le_83, unsqueeze_1690, le_84, unsqueeze_1702, unsqueeze_1714, le_86, unsqueeze_1726, le_87, unsqueeze_1738, le_88, unsqueeze_1750, le_89, unsqueeze_1762, unsqueeze_1774, le_91, unsqueeze_1786, le_92, unsqueeze_1798, le_93, unsqueeze_1810, le_94, unsqueeze_1822, unsqueeze_1834, le_96, unsqueeze_1846, le_97, unsqueeze_1858, le_98, unsqueeze_1870, le_99, unsqueeze_1882, unsqueeze_1894, le_101, unsqueeze_1906, le_102, unsqueeze_1918, le_103, unsqueeze_1930, le_104, unsqueeze_1942, unsqueeze_1954, le_106, unsqueeze_1966, le_107, unsqueeze_1978, le_108, unsqueeze_1990, le_109, unsqueeze_2002, unsqueeze_2014, le_111, unsqueeze_2026, le_112, unsqueeze_2038, le_113, unsqueeze_2050, le_114, unsqueeze_2062, unsqueeze_2074, le_116, unsqueeze_2086, le_117, unsqueeze_2098, le_118, unsqueeze_2110, le_119, unsqueeze_2122, unsqueeze_2134, le_121, unsqueeze_2146, le_122, unsqueeze_2158, le_123, unsqueeze_2170, le_124, unsqueeze_2182, unsqueeze_2194, unsqueeze_2206, le_126, unsqueeze_2218, le_127, unsqueeze_2230, le_128, unsqueeze_2242, le_129, unsqueeze_2254, unsqueeze_2266, le_131, unsqueeze_2278, le_132, unsqueeze_2290, le_133, unsqueeze_2302, le_134, unsqueeze_2314, unsqueeze_2326, le_136, unsqueeze_2338, le_137, unsqueeze_2350, le_138, unsqueeze_2362, le_139, unsqueeze_2374, unsqueeze_2386, le_141, unsqueeze_2398, le_142, unsqueeze_2410, le_143, unsqueeze_2422, le_144, unsqueeze_2434, unsqueeze_2446, unsqueeze_2458, le_146, unsqueeze_2470, le_147, unsqueeze_2482, le_148, unsqueeze_2494, le_149, unsqueeze_2506, unsqueeze_2518, le_151, unsqueeze_2530, le_152, unsqueeze_2542, le_153, unsqueeze_2554, le_154, unsqueeze_2566, unsqueeze_2578, le_156, unsqueeze_2590, le_157, unsqueeze_2602, le_158, unsqueeze_2614, le_159, unsqueeze_2626, unsqueeze_2638, unsqueeze_2650, le_161, unsqueeze_2662, le_162, unsqueeze_2674, le_163, unsqueeze_2686, le_164, unsqueeze_2698, unsqueeze_2710, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 1, 21, 3))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_4, (104, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_5, (104, ), (1, ))
    assert_size_stride(primals_7, (26, 26, 3, 3), (234, 1, 78, 26))
    assert_size_stride(primals_8, (26, ), (1, ))
    assert_size_stride(primals_10, (26, 26, 3, 3), (234, 1, 78, 26))
    assert_size_stride(primals_11, (26, ), (1, ))
    assert_size_stride(primals_13, (26, 26, 3, 3), (234, 1, 78, 26))
    assert_size_stride(primals_14, (26, ), (1, ))
    assert_size_stride(primals_16, (256, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_19, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_22, (104, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_23, (104, ), (1, ))
    assert_size_stride(primals_25, (26, 26, 3, 3), (234, 1, 78, 26))
    assert_size_stride(primals_26, (26, ), (1, ))
    assert_size_stride(primals_28, (26, 26, 3, 3), (234, 1, 78, 26))
    assert_size_stride(primals_29, (26, ), (1, ))
    assert_size_stride(primals_31, (26, 26, 3, 3), (234, 1, 78, 26))
    assert_size_stride(primals_32, (26, ), (1, ))
    assert_size_stride(primals_34, (256, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_37, (104, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_38, (104, ), (1, ))
    assert_size_stride(primals_40, (26, 26, 3, 3), (234, 1, 78, 26))
    assert_size_stride(primals_41, (26, ), (1, ))
    assert_size_stride(primals_43, (26, 26, 3, 3), (234, 1, 78, 26))
    assert_size_stride(primals_44, (26, ), (1, ))
    assert_size_stride(primals_46, (26, 26, 3, 3), (234, 1, 78, 26))
    assert_size_stride(primals_47, (26, ), (1, ))
    assert_size_stride(primals_49, (256, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_52, (208, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_53, (208, ), (1, ))
    assert_size_stride(primals_55, (52, 52, 3, 3), (468, 1, 156, 52))
    assert_size_stride(primals_56, (52, ), (1, ))
    assert_size_stride(primals_58, (52, 52, 3, 3), (468, 1, 156, 52))
    assert_size_stride(primals_59, (52, ), (1, ))
    assert_size_stride(primals_61, (52, 52, 3, 3), (468, 1, 156, 52))
    assert_size_stride(primals_62, (52, ), (1, ))
    assert_size_stride(primals_64, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_67, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_68, (512, ), (1, ))
    assert_size_stride(primals_70, (208, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_71, (208, ), (1, ))
    assert_size_stride(primals_73, (52, 52, 3, 3), (468, 1, 156, 52))
    assert_size_stride(primals_74, (52, ), (1, ))
    assert_size_stride(primals_76, (52, 52, 3, 3), (468, 1, 156, 52))
    assert_size_stride(primals_77, (52, ), (1, ))
    assert_size_stride(primals_79, (52, 52, 3, 3), (468, 1, 156, 52))
    assert_size_stride(primals_80, (52, ), (1, ))
    assert_size_stride(primals_82, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(primals_83, (512, ), (1, ))
    assert_size_stride(primals_85, (208, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_86, (208, ), (1, ))
    assert_size_stride(primals_88, (52, 52, 3, 3), (468, 1, 156, 52))
    assert_size_stride(primals_89, (52, ), (1, ))
    assert_size_stride(primals_91, (52, 52, 3, 3), (468, 1, 156, 52))
    assert_size_stride(primals_92, (52, ), (1, ))
    assert_size_stride(primals_94, (52, 52, 3, 3), (468, 1, 156, 52))
    assert_size_stride(primals_95, (52, ), (1, ))
    assert_size_stride(primals_97, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_100, (208, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_101, (208, ), (1, ))
    assert_size_stride(primals_103, (52, 52, 3, 3), (468, 1, 156, 52))
    assert_size_stride(primals_104, (52, ), (1, ))
    assert_size_stride(primals_106, (52, 52, 3, 3), (468, 1, 156, 52))
    assert_size_stride(primals_107, (52, ), (1, ))
    assert_size_stride(primals_109, (52, 52, 3, 3), (468, 1, 156, 52))
    assert_size_stride(primals_110, (52, ), (1, ))
    assert_size_stride(primals_112, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_115, (416, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_116, (416, ), (1, ))
    assert_size_stride(primals_118, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_119, (104, ), (1, ))
    assert_size_stride(primals_121, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_122, (104, ), (1, ))
    assert_size_stride(primals_124, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_125, (104, ), (1, ))
    assert_size_stride(primals_127, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_128, (1024, ), (1, ))
    assert_size_stride(primals_130, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_131, (1024, ), (1, ))
    assert_size_stride(primals_133, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_134, (416, ), (1, ))
    assert_size_stride(primals_136, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_137, (104, ), (1, ))
    assert_size_stride(primals_139, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_140, (104, ), (1, ))
    assert_size_stride(primals_142, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_143, (104, ), (1, ))
    assert_size_stride(primals_145, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_146, (1024, ), (1, ))
    assert_size_stride(primals_148, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_149, (416, ), (1, ))
    assert_size_stride(primals_151, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_152, (104, ), (1, ))
    assert_size_stride(primals_154, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_155, (104, ), (1, ))
    assert_size_stride(primals_157, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_158, (104, ), (1, ))
    assert_size_stride(primals_160, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_161, (1024, ), (1, ))
    assert_size_stride(primals_163, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_164, (416, ), (1, ))
    assert_size_stride(primals_166, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_167, (104, ), (1, ))
    assert_size_stride(primals_169, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_170, (104, ), (1, ))
    assert_size_stride(primals_172, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_173, (104, ), (1, ))
    assert_size_stride(primals_175, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_176, (1024, ), (1, ))
    assert_size_stride(primals_178, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_179, (416, ), (1, ))
    assert_size_stride(primals_181, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_182, (104, ), (1, ))
    assert_size_stride(primals_184, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_185, (104, ), (1, ))
    assert_size_stride(primals_187, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_188, (104, ), (1, ))
    assert_size_stride(primals_190, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_191, (1024, ), (1, ))
    assert_size_stride(primals_193, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_194, (416, ), (1, ))
    assert_size_stride(primals_196, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_197, (104, ), (1, ))
    assert_size_stride(primals_199, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_200, (104, ), (1, ))
    assert_size_stride(primals_202, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_203, (104, ), (1, ))
    assert_size_stride(primals_205, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_206, (1024, ), (1, ))
    assert_size_stride(primals_208, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_209, (416, ), (1, ))
    assert_size_stride(primals_211, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_212, (104, ), (1, ))
    assert_size_stride(primals_214, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_215, (104, ), (1, ))
    assert_size_stride(primals_217, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_218, (104, ), (1, ))
    assert_size_stride(primals_220, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_221, (1024, ), (1, ))
    assert_size_stride(primals_223, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_224, (416, ), (1, ))
    assert_size_stride(primals_226, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_227, (104, ), (1, ))
    assert_size_stride(primals_229, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_230, (104, ), (1, ))
    assert_size_stride(primals_232, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_233, (104, ), (1, ))
    assert_size_stride(primals_235, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_236, (1024, ), (1, ))
    assert_size_stride(primals_238, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_239, (416, ), (1, ))
    assert_size_stride(primals_241, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_242, (104, ), (1, ))
    assert_size_stride(primals_244, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_245, (104, ), (1, ))
    assert_size_stride(primals_247, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_248, (104, ), (1, ))
    assert_size_stride(primals_250, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_251, (1024, ), (1, ))
    assert_size_stride(primals_253, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_254, (416, ), (1, ))
    assert_size_stride(primals_256, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_257, (104, ), (1, ))
    assert_size_stride(primals_259, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_260, (104, ), (1, ))
    assert_size_stride(primals_262, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_263, (104, ), (1, ))
    assert_size_stride(primals_265, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_266, (1024, ), (1, ))
    assert_size_stride(primals_268, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_269, (416, ), (1, ))
    assert_size_stride(primals_271, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_272, (104, ), (1, ))
    assert_size_stride(primals_274, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_275, (104, ), (1, ))
    assert_size_stride(primals_277, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_278, (104, ), (1, ))
    assert_size_stride(primals_280, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_281, (1024, ), (1, ))
    assert_size_stride(primals_283, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_284, (416, ), (1, ))
    assert_size_stride(primals_286, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_287, (104, ), (1, ))
    assert_size_stride(primals_289, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_290, (104, ), (1, ))
    assert_size_stride(primals_292, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_293, (104, ), (1, ))
    assert_size_stride(primals_295, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_296, (1024, ), (1, ))
    assert_size_stride(primals_298, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_299, (416, ), (1, ))
    assert_size_stride(primals_301, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_302, (104, ), (1, ))
    assert_size_stride(primals_304, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_305, (104, ), (1, ))
    assert_size_stride(primals_307, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_308, (104, ), (1, ))
    assert_size_stride(primals_310, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_311, (1024, ), (1, ))
    assert_size_stride(primals_313, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_314, (416, ), (1, ))
    assert_size_stride(primals_316, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_317, (104, ), (1, ))
    assert_size_stride(primals_319, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_320, (104, ), (1, ))
    assert_size_stride(primals_322, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_323, (104, ), (1, ))
    assert_size_stride(primals_325, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_326, (1024, ), (1, ))
    assert_size_stride(primals_328, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_329, (416, ), (1, ))
    assert_size_stride(primals_331, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_332, (104, ), (1, ))
    assert_size_stride(primals_334, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_335, (104, ), (1, ))
    assert_size_stride(primals_337, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_338, (104, ), (1, ))
    assert_size_stride(primals_340, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_341, (1024, ), (1, ))
    assert_size_stride(primals_343, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_344, (416, ), (1, ))
    assert_size_stride(primals_346, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_347, (104, ), (1, ))
    assert_size_stride(primals_349, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_350, (104, ), (1, ))
    assert_size_stride(primals_352, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_353, (104, ), (1, ))
    assert_size_stride(primals_355, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_356, (1024, ), (1, ))
    assert_size_stride(primals_358, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_359, (416, ), (1, ))
    assert_size_stride(primals_361, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_362, (104, ), (1, ))
    assert_size_stride(primals_364, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_365, (104, ), (1, ))
    assert_size_stride(primals_367, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_368, (104, ), (1, ))
    assert_size_stride(primals_370, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_371, (1024, ), (1, ))
    assert_size_stride(primals_373, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_374, (416, ), (1, ))
    assert_size_stride(primals_376, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_377, (104, ), (1, ))
    assert_size_stride(primals_379, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_380, (104, ), (1, ))
    assert_size_stride(primals_382, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_383, (104, ), (1, ))
    assert_size_stride(primals_385, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_386, (1024, ), (1, ))
    assert_size_stride(primals_388, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_389, (416, ), (1, ))
    assert_size_stride(primals_391, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_392, (104, ), (1, ))
    assert_size_stride(primals_394, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_395, (104, ), (1, ))
    assert_size_stride(primals_397, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_398, (104, ), (1, ))
    assert_size_stride(primals_400, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_401, (1024, ), (1, ))
    assert_size_stride(primals_403, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_404, (416, ), (1, ))
    assert_size_stride(primals_406, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_407, (104, ), (1, ))
    assert_size_stride(primals_409, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_410, (104, ), (1, ))
    assert_size_stride(primals_412, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_413, (104, ), (1, ))
    assert_size_stride(primals_415, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_416, (1024, ), (1, ))
    assert_size_stride(primals_418, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_419, (416, ), (1, ))
    assert_size_stride(primals_421, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_422, (104, ), (1, ))
    assert_size_stride(primals_424, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_425, (104, ), (1, ))
    assert_size_stride(primals_427, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_428, (104, ), (1, ))
    assert_size_stride(primals_430, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_431, (1024, ), (1, ))
    assert_size_stride(primals_433, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_434, (416, ), (1, ))
    assert_size_stride(primals_436, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_437, (104, ), (1, ))
    assert_size_stride(primals_439, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_440, (104, ), (1, ))
    assert_size_stride(primals_442, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_443, (104, ), (1, ))
    assert_size_stride(primals_445, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_446, (1024, ), (1, ))
    assert_size_stride(primals_448, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_449, (416, ), (1, ))
    assert_size_stride(primals_451, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_452, (104, ), (1, ))
    assert_size_stride(primals_454, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_455, (104, ), (1, ))
    assert_size_stride(primals_457, (104, 104, 3, 3), (936, 1, 312, 104))
    assert_size_stride(primals_458, (104, ), (1, ))
    assert_size_stride(primals_460, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_461, (1024, ), (1, ))
    assert_size_stride(primals_463, (832, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_464, (832, ), (1, ))
    assert_size_stride(primals_466, (208, 208, 3, 3), (1872, 1, 624, 208))
    assert_size_stride(primals_467, (208, ), (1, ))
    assert_size_stride(primals_469, (208, 208, 3, 3), (1872, 1, 624, 208))
    assert_size_stride(primals_470, (208, ), (1, ))
    assert_size_stride(primals_472, (208, 208, 3, 3), (1872, 1, 624, 208))
    assert_size_stride(primals_473, (208, ), (1, ))
    assert_size_stride(primals_475, (2048, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(primals_476, (2048, ), (1, ))
    assert_size_stride(primals_478, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_479, (2048, ), (1, ))
    assert_size_stride(primals_481, (832, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_482, (832, ), (1, ))
    assert_size_stride(primals_484, (208, 208, 3, 3), (1872, 1, 624, 208))
    assert_size_stride(primals_485, (208, ), (1, ))
    assert_size_stride(primals_487, (208, 208, 3, 3), (1872, 1, 624, 208))
    assert_size_stride(primals_488, (208, ), (1, ))
    assert_size_stride(primals_490, (208, 208, 3, 3), (1872, 1, 624, 208))
    assert_size_stride(primals_491, (208, ), (1, ))
    assert_size_stride(primals_493, (2048, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(primals_494, (2048, ), (1, ))
    assert_size_stride(primals_496, (832, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_497, (832, ), (1, ))
    assert_size_stride(primals_499, (208, 208, 3, 3), (1872, 1, 624, 208))
    assert_size_stride(primals_500, (208, ), (1, ))
    assert_size_stride(primals_502, (208, 208, 3, 3), (1872, 1, 624, 208))
    assert_size_stride(primals_503, (208, ), (1, ))
    assert_size_stride(primals_505, (208, 208, 3, 3), (1872, 1, 624, 208))
    assert_size_stride(primals_506, (208, ), (1, ))
    assert_size_stride(primals_508, (2048, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(primals_509, (2048, ), (1, ))
    assert_size_stride(primals_1023, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_1, (64, ), (1, ))
    assert_size_stride(relu, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(getitem_2, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(getitem_3, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_1, (8, 104, 56, 56), (326144, 1, 5824, 104))
    assert_size_stride(squeeze_4, (104, ), (1, ))
    assert_size_stride(getitem_10, (8, 26, 56, 56), (326144, 1, 5824, 104))
    assert_size_stride(convolution_2, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(squeeze_7, (26, ), (1, ))
    assert_size_stride(getitem_17, (8, 26, 56, 56), (326144, 1, 5824, 104))
    assert_size_stride(convolution_3, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(squeeze_10, (26, ), (1, ))
    assert_size_stride(getitem_24, (8, 26, 56, 56), (326144, 1, 5824, 104))
    assert_size_stride(convolution_4, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(squeeze_13, (26, ), (1, ))
    assert_size_stride(getitem_31, (8, 26, 56, 56), (326144, 1, 5824, 104))
    assert_size_stride(cat, (8, 104, 56, 56), (326144, 1, 5824, 104))
    assert_size_stride(convolution_5, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_16, (256, ), (1, ))
    assert_size_stride(convolution_6, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_19, (256, ), (1, ))
    assert_size_stride(relu_5, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_7, (8, 104, 56, 56), (326144, 1, 5824, 104))
    assert_size_stride(squeeze_22, (104, ), (1, ))
    assert_size_stride(getitem_42, (8, 26, 56, 56), (326144, 1, 5824, 104))
    assert_size_stride(convolution_8, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(squeeze_25, (26, ), (1, ))
    assert_size_stride(add_46, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(convolution_9, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(squeeze_28, (26, ), (1, ))
    assert_size_stride(add_52, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(convolution_10, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(squeeze_31, (26, ), (1, ))
    assert_size_stride(cat_1, (8, 104, 56, 56), (326144, 1, 5824, 104))
    assert_size_stride(convolution_11, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_34, (256, ), (1, ))
    assert_size_stride(relu_10, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_12, (8, 104, 56, 56), (326144, 1, 5824, 104))
    assert_size_stride(squeeze_37, (104, ), (1, ))
    assert_size_stride(getitem_72, (8, 26, 56, 56), (326144, 1, 5824, 104))
    assert_size_stride(convolution_13, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(squeeze_40, (26, ), (1, ))
    assert_size_stride(add_74, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(convolution_14, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(squeeze_43, (26, ), (1, ))
    assert_size_stride(add_80, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(convolution_15, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(squeeze_46, (26, ), (1, ))
    assert_size_stride(cat_2, (8, 104, 56, 56), (326144, 1, 5824, 104))
    assert_size_stride(convolution_16, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_49, (256, ), (1, ))
    assert_size_stride(relu_15, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_17, (8, 208, 56, 56), (652288, 1, 11648, 208))
    assert_size_stride(squeeze_52, (208, ), (1, ))
    assert_size_stride(getitem_102, (8, 52, 56, 56), (652288, 1, 11648, 208))
    assert_size_stride(convolution_18, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(squeeze_55, (52, ), (1, ))
    assert_size_stride(getitem_109, (8, 52, 56, 56), (652288, 1, 11648, 208))
    assert_size_stride(convolution_19, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(squeeze_58, (52, ), (1, ))
    assert_size_stride(getitem_116, (8, 52, 56, 56), (652288, 1, 11648, 208))
    assert_size_stride(convolution_20, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(squeeze_61, (52, ), (1, ))
    assert_size_stride(getitem_123, (8, 52, 56, 56), (652288, 1, 11648, 208))
    assert_size_stride(cat_3, (8, 208, 28, 28), (163072, 1, 5824, 208))
    assert_size_stride(convolution_21, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_64, (512, ), (1, ))
    assert_size_stride(convolution_22, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_67, (512, ), (1, ))
    assert_size_stride(relu_20, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_23, (8, 208, 28, 28), (163072, 1, 5824, 208))
    assert_size_stride(squeeze_70, (208, ), (1, ))
    assert_size_stride(getitem_134, (8, 52, 28, 28), (163072, 1, 5824, 208))
    assert_size_stride(convolution_24, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(squeeze_73, (52, ), (1, ))
    assert_size_stride(add_133, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(convolution_25, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(squeeze_76, (52, ), (1, ))
    assert_size_stride(add_139, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(convolution_26, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(squeeze_79, (52, ), (1, ))
    assert_size_stride(cat_4, (8, 208, 28, 28), (163072, 1, 5824, 208))
    assert_size_stride(convolution_27, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_82, (512, ), (1, ))
    assert_size_stride(relu_25, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_28, (8, 208, 28, 28), (163072, 1, 5824, 208))
    assert_size_stride(squeeze_85, (208, ), (1, ))
    assert_size_stride(getitem_164, (8, 52, 28, 28), (163072, 1, 5824, 208))
    assert_size_stride(convolution_29, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(squeeze_88, (52, ), (1, ))
    assert_size_stride(add_161, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(convolution_30, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(squeeze_91, (52, ), (1, ))
    assert_size_stride(add_167, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(convolution_31, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(squeeze_94, (52, ), (1, ))
    assert_size_stride(cat_5, (8, 208, 28, 28), (163072, 1, 5824, 208))
    assert_size_stride(convolution_32, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_97, (512, ), (1, ))
    assert_size_stride(relu_30, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_33, (8, 208, 28, 28), (163072, 1, 5824, 208))
    assert_size_stride(squeeze_100, (208, ), (1, ))
    assert_size_stride(getitem_194, (8, 52, 28, 28), (163072, 1, 5824, 208))
    assert_size_stride(convolution_34, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(squeeze_103, (52, ), (1, ))
    assert_size_stride(add_189, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(convolution_35, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(squeeze_106, (52, ), (1, ))
    assert_size_stride(add_195, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(convolution_36, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(squeeze_109, (52, ), (1, ))
    assert_size_stride(cat_6, (8, 208, 28, 28), (163072, 1, 5824, 208))
    assert_size_stride(convolution_37, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_112, (512, ), (1, ))
    assert_size_stride(relu_35, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_38, (8, 416, 28, 28), (326144, 1, 11648, 416))
    assert_size_stride(squeeze_115, (416, ), (1, ))
    assert_size_stride(getitem_224, (8, 104, 28, 28), (326144, 1, 11648, 416))
    assert_size_stride(convolution_39, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_118, (104, ), (1, ))
    assert_size_stride(getitem_231, (8, 104, 28, 28), (326144, 1, 11648, 416))
    assert_size_stride(convolution_40, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_121, (104, ), (1, ))
    assert_size_stride(getitem_238, (8, 104, 28, 28), (326144, 1, 11648, 416))
    assert_size_stride(convolution_41, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_124, (104, ), (1, ))
    assert_size_stride(getitem_245, (8, 104, 28, 28), (326144, 1, 11648, 416))
    assert_size_stride(cat_7, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_42, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_127, (1024, ), (1, ))
    assert_size_stride(convolution_43, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_130, (1024, ), (1, ))
    assert_size_stride(relu_40, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_44, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_133, (416, ), (1, ))
    assert_size_stride(getitem_256, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_45, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_136, (104, ), (1, ))
    assert_size_stride(add_248, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_46, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_139, (104, ), (1, ))
    assert_size_stride(add_254, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_47, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_142, (104, ), (1, ))
    assert_size_stride(cat_8, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_48, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_145, (1024, ), (1, ))
    assert_size_stride(relu_45, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_49, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_148, (416, ), (1, ))
    assert_size_stride(getitem_286, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_50, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_151, (104, ), (1, ))
    assert_size_stride(add_276, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_51, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_154, (104, ), (1, ))
    assert_size_stride(add_282, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_52, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_157, (104, ), (1, ))
    assert_size_stride(cat_9, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_53, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_160, (1024, ), (1, ))
    assert_size_stride(relu_50, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_54, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_163, (416, ), (1, ))
    assert_size_stride(getitem_316, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_55, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_166, (104, ), (1, ))
    assert_size_stride(add_304, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_56, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_169, (104, ), (1, ))
    assert_size_stride(add_310, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_57, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_172, (104, ), (1, ))
    assert_size_stride(cat_10, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_58, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_175, (1024, ), (1, ))
    assert_size_stride(relu_55, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_59, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_178, (416, ), (1, ))
    assert_size_stride(getitem_346, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_60, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_181, (104, ), (1, ))
    assert_size_stride(add_332, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_61, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_184, (104, ), (1, ))
    assert_size_stride(add_338, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_62, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_187, (104, ), (1, ))
    assert_size_stride(cat_11, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_63, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_190, (1024, ), (1, ))
    assert_size_stride(relu_60, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_64, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_193, (416, ), (1, ))
    assert_size_stride(getitem_376, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_65, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_196, (104, ), (1, ))
    assert_size_stride(add_360, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_66, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_199, (104, ), (1, ))
    assert_size_stride(add_366, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_67, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_202, (104, ), (1, ))
    assert_size_stride(cat_12, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_68, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_205, (1024, ), (1, ))
    assert_size_stride(relu_65, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_69, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_208, (416, ), (1, ))
    assert_size_stride(getitem_406, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_70, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_211, (104, ), (1, ))
    assert_size_stride(add_388, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_71, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_214, (104, ), (1, ))
    assert_size_stride(add_394, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_72, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_217, (104, ), (1, ))
    assert_size_stride(cat_13, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_73, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_220, (1024, ), (1, ))
    assert_size_stride(relu_70, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_74, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_223, (416, ), (1, ))
    assert_size_stride(getitem_436, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_75, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_226, (104, ), (1, ))
    assert_size_stride(add_416, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_76, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_229, (104, ), (1, ))
    assert_size_stride(add_422, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_77, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_232, (104, ), (1, ))
    assert_size_stride(cat_14, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_78, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_235, (1024, ), (1, ))
    assert_size_stride(relu_75, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_79, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_238, (416, ), (1, ))
    assert_size_stride(getitem_466, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_80, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_241, (104, ), (1, ))
    assert_size_stride(add_444, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_81, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_244, (104, ), (1, ))
    assert_size_stride(add_450, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_82, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_247, (104, ), (1, ))
    assert_size_stride(cat_15, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_83, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_250, (1024, ), (1, ))
    assert_size_stride(relu_80, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_84, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_253, (416, ), (1, ))
    assert_size_stride(getitem_496, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_85, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_256, (104, ), (1, ))
    assert_size_stride(add_472, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_86, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_259, (104, ), (1, ))
    assert_size_stride(add_478, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_87, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_262, (104, ), (1, ))
    assert_size_stride(cat_16, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_88, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_265, (1024, ), (1, ))
    assert_size_stride(relu_85, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_89, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_268, (416, ), (1, ))
    assert_size_stride(getitem_526, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_90, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_271, (104, ), (1, ))
    assert_size_stride(add_500, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_91, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_274, (104, ), (1, ))
    assert_size_stride(add_506, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_92, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_277, (104, ), (1, ))
    assert_size_stride(cat_17, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_93, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_280, (1024, ), (1, ))
    assert_size_stride(relu_90, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_94, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_283, (416, ), (1, ))
    assert_size_stride(getitem_556, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_95, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_286, (104, ), (1, ))
    assert_size_stride(add_528, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_96, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_289, (104, ), (1, ))
    assert_size_stride(add_534, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_97, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_292, (104, ), (1, ))
    assert_size_stride(cat_18, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_98, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_295, (1024, ), (1, ))
    assert_size_stride(relu_95, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_99, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_298, (416, ), (1, ))
    assert_size_stride(getitem_586, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_100, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_301, (104, ), (1, ))
    assert_size_stride(add_556, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_101, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_304, (104, ), (1, ))
    assert_size_stride(add_562, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_102, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_307, (104, ), (1, ))
    assert_size_stride(cat_19, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_103, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_310, (1024, ), (1, ))
    assert_size_stride(relu_100, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_104, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_313, (416, ), (1, ))
    assert_size_stride(getitem_616, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_105, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_316, (104, ), (1, ))
    assert_size_stride(add_584, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_106, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_319, (104, ), (1, ))
    assert_size_stride(add_590, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_107, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_322, (104, ), (1, ))
    assert_size_stride(cat_20, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_108, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_325, (1024, ), (1, ))
    assert_size_stride(relu_105, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_109, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_328, (416, ), (1, ))
    assert_size_stride(getitem_646, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_110, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_331, (104, ), (1, ))
    assert_size_stride(add_612, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_111, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_334, (104, ), (1, ))
    assert_size_stride(add_618, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_112, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_337, (104, ), (1, ))
    assert_size_stride(cat_21, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_113, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_340, (1024, ), (1, ))
    assert_size_stride(relu_110, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_114, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_343, (416, ), (1, ))
    assert_size_stride(getitem_676, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_115, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_346, (104, ), (1, ))
    assert_size_stride(add_640, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_116, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_349, (104, ), (1, ))
    assert_size_stride(add_646, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_117, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_352, (104, ), (1, ))
    assert_size_stride(cat_22, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_118, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_355, (1024, ), (1, ))
    assert_size_stride(relu_115, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_119, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_358, (416, ), (1, ))
    assert_size_stride(getitem_706, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_120, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_361, (104, ), (1, ))
    assert_size_stride(add_668, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_121, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_364, (104, ), (1, ))
    assert_size_stride(add_674, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_122, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_367, (104, ), (1, ))
    assert_size_stride(cat_23, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_123, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_370, (1024, ), (1, ))
    assert_size_stride(relu_120, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_124, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_373, (416, ), (1, ))
    assert_size_stride(getitem_736, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_125, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_376, (104, ), (1, ))
    assert_size_stride(add_696, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_126, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_379, (104, ), (1, ))
    assert_size_stride(add_702, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_127, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_382, (104, ), (1, ))
    assert_size_stride(cat_24, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_128, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_385, (1024, ), (1, ))
    assert_size_stride(relu_125, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_129, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_388, (416, ), (1, ))
    assert_size_stride(getitem_766, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_130, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_391, (104, ), (1, ))
    assert_size_stride(add_724, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_131, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_394, (104, ), (1, ))
    assert_size_stride(add_730, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_132, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_397, (104, ), (1, ))
    assert_size_stride(cat_25, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_133, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_400, (1024, ), (1, ))
    assert_size_stride(relu_130, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_134, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_403, (416, ), (1, ))
    assert_size_stride(getitem_796, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_135, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_406, (104, ), (1, ))
    assert_size_stride(add_752, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_136, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_409, (104, ), (1, ))
    assert_size_stride(add_758, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_137, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_412, (104, ), (1, ))
    assert_size_stride(cat_26, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_138, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_415, (1024, ), (1, ))
    assert_size_stride(relu_135, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_139, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_418, (416, ), (1, ))
    assert_size_stride(getitem_826, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_140, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_421, (104, ), (1, ))
    assert_size_stride(add_780, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_141, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_424, (104, ), (1, ))
    assert_size_stride(add_786, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_142, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_427, (104, ), (1, ))
    assert_size_stride(cat_27, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_143, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_430, (1024, ), (1, ))
    assert_size_stride(relu_140, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_144, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_433, (416, ), (1, ))
    assert_size_stride(getitem_856, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_145, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_436, (104, ), (1, ))
    assert_size_stride(add_808, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_146, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_439, (104, ), (1, ))
    assert_size_stride(add_814, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_147, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_442, (104, ), (1, ))
    assert_size_stride(cat_28, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_148, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_445, (1024, ), (1, ))
    assert_size_stride(relu_145, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_149, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(squeeze_448, (416, ), (1, ))
    assert_size_stride(getitem_886, (8, 104, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_150, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_451, (104, ), (1, ))
    assert_size_stride(add_836, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_151, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_454, (104, ), (1, ))
    assert_size_stride(add_842, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_152, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_457, (104, ), (1, ))
    assert_size_stride(cat_29, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(convolution_153, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_460, (1024, ), (1, ))
    assert_size_stride(relu_150, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_154, (8, 832, 14, 14), (163072, 1, 11648, 832))
    assert_size_stride(squeeze_463, (832, ), (1, ))
    assert_size_stride(getitem_916, (8, 208, 14, 14), (163072, 1, 11648, 832))
    assert_size_stride(convolution_155, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(squeeze_466, (208, ), (1, ))
    assert_size_stride(getitem_923, (8, 208, 14, 14), (163072, 1, 11648, 832))
    assert_size_stride(convolution_156, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(squeeze_469, (208, ), (1, ))
    assert_size_stride(getitem_930, (8, 208, 14, 14), (163072, 1, 11648, 832))
    assert_size_stride(convolution_157, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(squeeze_472, (208, ), (1, ))
    assert_size_stride(getitem_937, (8, 208, 14, 14), (163072, 1, 11648, 832))
    assert_size_stride(cat_30, (8, 832, 7, 7), (40768, 1, 5824, 832))
    assert_size_stride(convolution_158, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_475, (2048, ), (1, ))
    assert_size_stride(convolution_159, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_478, (2048, ), (1, ))
    assert_size_stride(relu_155, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(convolution_160, (8, 832, 7, 7), (40768, 1, 5824, 832))
    assert_size_stride(squeeze_481, (832, ), (1, ))
    assert_size_stride(getitem_948, (8, 208, 7, 7), (40768, 1, 5824, 832))
    assert_size_stride(convolution_161, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(squeeze_484, (208, ), (1, ))
    assert_size_stride(add_895, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(convolution_162, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(squeeze_487, (208, ), (1, ))
    assert_size_stride(add_901, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(convolution_163, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(squeeze_490, (208, ), (1, ))
    assert_size_stride(cat_31, (8, 832, 7, 7), (40768, 1, 5824, 832))
    assert_size_stride(convolution_164, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_493, (2048, ), (1, ))
    assert_size_stride(relu_160, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(convolution_165, (8, 832, 7, 7), (40768, 1, 5824, 832))
    assert_size_stride(squeeze_496, (832, ), (1, ))
    assert_size_stride(getitem_978, (8, 208, 7, 7), (40768, 1, 5824, 832))
    assert_size_stride(convolution_166, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(squeeze_499, (208, ), (1, ))
    assert_size_stride(add_923, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(convolution_167, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(squeeze_502, (208, ), (1, ))
    assert_size_stride(add_929, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(convolution_168, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(squeeze_505, (208, ), (1, ))
    assert_size_stride(cat_32, (8, 832, 7, 7), (40768, 1, 5824, 832))
    assert_size_stride(convolution_169, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_508, (2048, ), (1, ))
    assert_size_stride(view, (8, 2048), (2048, 1))
    assert_size_stride(permute_1, (1000, 2048), (2048, 1))
    assert_size_stride(le, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(unsqueeze_682, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(le_1, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(unsqueeze_694, (1, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(le_2, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(unsqueeze_706, (1, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(le_3, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(unsqueeze_718, (1, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(le_4, (8, 832, 7, 7), (40768, 1, 5824, 832))
    assert_size_stride(unsqueeze_730, (1, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(unsqueeze_742, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(le_6, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(unsqueeze_754, (1, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(le_7, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(unsqueeze_766, (1, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(le_8, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(unsqueeze_778, (1, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(le_9, (8, 832, 7, 7), (40768, 1, 5824, 832))
    assert_size_stride(unsqueeze_790, (1, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(unsqueeze_802, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_814, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(le_11, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(unsqueeze_826, (1, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(le_12, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(unsqueeze_838, (1, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(le_13, (8, 208, 7, 7), (10192, 1, 1456, 208))
    assert_size_stride(unsqueeze_850, (1, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(le_14, (8, 832, 14, 14), (163072, 1, 11648, 832))
    assert_size_stride(unsqueeze_862, (1, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(unsqueeze_874, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_16, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_886, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_17, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_898, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_18, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_910, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_19, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_922, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_934, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_21, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_946, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_22, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_958, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_23, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_970, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_24, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_982, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_994, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_26, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1006, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_27, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1018, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_28, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1030, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_29, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_1042, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_1054, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_31, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1066, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_32, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1078, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_33, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1090, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_34, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_1102, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_1114, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_36, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1126, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_37, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1138, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_38, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1150, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_39, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_1162, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_1174, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_41, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1186, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_42, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1198, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_43, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1210, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_44, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_1222, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_1234, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_46, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1246, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_47, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1258, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_48, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1270, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_49, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_1282, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_1294, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_51, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1306, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_52, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1318, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_53, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1330, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_54, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_1342, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_1354, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_56, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1366, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_57, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1378, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_58, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1390, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_59, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_1402, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_1414, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_61, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1426, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_62, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1438, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_63, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1450, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_64, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_1462, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_1474, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_66, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1486, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_67, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1498, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_68, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1510, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_69, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_1522, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_1534, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_71, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1546, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_72, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1558, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_73, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1570, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_74, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_1582, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_1594, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_76, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1606, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_77, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1618, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_78, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1630, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_79, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_1642, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_1654, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_81, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1666, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_82, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1678, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_83, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1690, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_84, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_1702, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_1714, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_86, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1726, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_87, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1738, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_88, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1750, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_89, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_1762, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_1774, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_91, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1786, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_92, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1798, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_93, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1810, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_94, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_1822, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_1834, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_96, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1846, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_97, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1858, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_98, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1870, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_99, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_1882, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_1894, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_101, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1906, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_102, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1918, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_103, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1930, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_104, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_1942, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_1954, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_106, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1966, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_107, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1978, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_108, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_1990, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_109, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_2002, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_2014, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_111, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_2026, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_112, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_2038, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_113, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_2050, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_114, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_2062, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_2074, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_116, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_2086, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_117, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_2098, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_118, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_2110, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_119, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_2122, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_2134, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_121, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_2146, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_122, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_2158, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_123, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_2170, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_124, (8, 416, 14, 14), (81536, 1, 5824, 416))
    assert_size_stride(unsqueeze_2182, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_2194, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_2206, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_126, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_2218, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_127, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_2230, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_128, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(unsqueeze_2242, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(le_129, (8, 416, 28, 28), (326144, 1, 11648, 416))
    assert_size_stride(unsqueeze_2254, (1, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(unsqueeze_2266, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_131, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(unsqueeze_2278, (1, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(le_132, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(unsqueeze_2290, (1, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(le_133, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(unsqueeze_2302, (1, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(le_134, (8, 208, 28, 28), (163072, 1, 5824, 208))
    assert_size_stride(unsqueeze_2314, (1, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(unsqueeze_2326, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_136, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(unsqueeze_2338, (1, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(le_137, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(unsqueeze_2350, (1, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(le_138, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(unsqueeze_2362, (1, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(le_139, (8, 208, 28, 28), (163072, 1, 5824, 208))
    assert_size_stride(unsqueeze_2374, (1, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(unsqueeze_2386, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_141, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(unsqueeze_2398, (1, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(le_142, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(unsqueeze_2410, (1, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(le_143, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(unsqueeze_2422, (1, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(le_144, (8, 208, 28, 28), (163072, 1, 5824, 208))
    assert_size_stride(unsqueeze_2434, (1, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(unsqueeze_2446, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_2458, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_146, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(unsqueeze_2470, (1, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(le_147, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(unsqueeze_2482, (1, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(le_148, (8, 52, 28, 28), (40768, 1, 1456, 52))
    assert_size_stride(unsqueeze_2494, (1, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(le_149, (8, 208, 56, 56), (652288, 1, 11648, 208))
    assert_size_stride(unsqueeze_2506, (1, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(unsqueeze_2518, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_151, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(unsqueeze_2530, (1, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(le_152, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(unsqueeze_2542, (1, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(le_153, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(unsqueeze_2554, (1, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(le_154, (8, 104, 56, 56), (326144, 1, 5824, 104))
    assert_size_stride(unsqueeze_2566, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(unsqueeze_2578, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_156, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(unsqueeze_2590, (1, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(le_157, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(unsqueeze_2602, (1, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(le_158, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(unsqueeze_2614, (1, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(le_159, (8, 104, 56, 56), (326144, 1, 5824, 104))
    assert_size_stride(unsqueeze_2626, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(unsqueeze_2638, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_2650, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_161, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(unsqueeze_2662, (1, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(le_162, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(unsqueeze_2674, (1, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(le_163, (8, 26, 56, 56), (81536, 1, 1456, 26))
    assert_size_stride(unsqueeze_2686, (1, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(le_164, (8, 104, 56, 56), (326144, 1, 5824, 104))
    assert_size_stride(unsqueeze_2698, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(unsqueeze_2710, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), view, out=buf1)
        del view
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty_strided((2048, 4), (1, 2048), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((2048, 4), (1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_div_native_batch_norm_backward_threshold_backward_1.run(le, buf0, convolution_169, unsqueeze_682, buf3, buf5, 8192, 98, grid=grid(8192), stream=stream0)
        buf4 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_2.run(buf3, buf4, 2048, 4, grid=grid(2048), stream=stream0)
        buf6 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf7 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_3.run(buf5, squeeze_508, buf6, buf7, 2048, 4, grid=grid(2048), stream=stream0)
        buf8 = empty_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4.run(le, buf0, convolution_169, unsqueeze_682, buf6, squeeze_508, buf4, primals_509, buf8, 802816, grid=grid(802816), stream=stream0)
        del convolution_169
        del primals_509
        del squeeze_508
        del unsqueeze_682
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf9 = aten.convolution_backward(buf8, cat_32, primals_508, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_32
        del primals_508
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        buf12 = empty_strided((208, 4), (1, 208), device='cuda', dtype=torch.float32)
        buf14 = empty_strided((208, 4), (1, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(le_1, buf10, convolution_168, unsqueeze_694, buf12, buf14, 832, 98, grid=grid(832), stream=stream0)
        buf13 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(buf12, buf13, 208, 4, grid=grid(208), stream=stream0)
        buf15 = empty((208, ), device='cuda', dtype=torch.float32)
        buf16 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf14, squeeze_505, buf15, buf16, 208, 4, grid=grid(208), stream=stream0)
        buf17 = empty_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8.run(le_1, buf10, convolution_168, unsqueeze_694, buf15, squeeze_505, buf13, primals_506, buf17, 392, 208, grid=grid(392, 208), stream=stream0)
        del convolution_168
        del le_1
        del primals_506
        del squeeze_505
        del unsqueeze_694
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf18 = aten.convolution_backward(buf17, add_929, primals_505, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_929
        del primals_505
        buf19 = buf18[0]
        buf20 = buf18[1]
        del buf18
        buf21 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_9.run(le_2, buf10, buf19, buf21, 208, 392, grid=grid(208), stream=stream0)
        buf22 = buf14; del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_10.run(le_2, buf10, buf19, convolution_167, unsqueeze_706, buf22, 832, 98, grid=grid(832), stream=stream0)
        buf23 = empty((208, ), device='cuda', dtype=torch.float32)
        buf25 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf22, squeeze_502, buf23, buf25, 208, 4, grid=grid(208), stream=stream0)
        buf24 = buf17; del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_11.run(le_2, buf10, buf19, convolution_167, unsqueeze_706, buf23, squeeze_502, buf21, primals_503, buf24, 392, 208, grid=grid(392, 208), stream=stream0)
        del convolution_167
        del le_2
        del primals_503
        del squeeze_502
        del unsqueeze_706
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf26 = aten.convolution_backward(buf24, add_923, primals_502, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_923
        del primals_502
        buf27 = buf26[0]
        buf28 = buf26[1]
        del buf26
        buf29 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_12.run(le_3, buf10, buf27, buf29, 208, 392, grid=grid(208), stream=stream0)
        buf30 = buf22; del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_13.run(le_3, buf10, buf27, convolution_166, unsqueeze_718, buf30, 832, 98, grid=grid(832), stream=stream0)
        buf31 = empty((208, ), device='cuda', dtype=torch.float32)
        buf33 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf30, squeeze_499, buf31, buf33, 208, 4, grid=grid(208), stream=stream0)
        buf32 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_14.run(le_3, buf10, buf27, convolution_166, unsqueeze_718, buf31, squeeze_499, buf29, primals_500, buf32, 392, 208, grid=grid(392, 208), stream=stream0)
        del convolution_166
        del le_3
        del primals_500
        del squeeze_499
        del unsqueeze_718
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf34 = aten.convolution_backward(buf32, getitem_978, primals_499, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf32
        del getitem_978
        del primals_499
        buf35 = buf34[0]
        buf36 = buf34[1]
        del buf34
        buf37 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_15.run(buf37, le_4, buf35, buf27, buf19, 6656, 49, grid=grid(6656, 49), stream=stream0)
        del buf19
        del buf27
        del le_4
        buf38 = reinterpret_tensor(buf30, (832, ), (1, ), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_16.run(buf37, buf38, 832, 392, grid=grid(832), stream=stream0)
        buf39 = empty_strided((832, 4), (1, 832), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_17.run(buf37, convolution_165, unsqueeze_730, buf39, 3328, 98, grid=grid(3328), stream=stream0)
        buf40 = reinterpret_tensor(buf12, (832, ), (1, ), 0); del buf12  # reuse
        buf41 = empty((832, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_18.run(buf39, squeeze_496, buf40, buf41, 832, 4, grid=grid(832), stream=stream0)
        buf42 = empty_strided((8, 832, 7, 7), (40768, 1, 5824, 832), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_19.run(buf37, convolution_165, unsqueeze_730, buf40, squeeze_496, buf38, primals_497, buf42, 392, 832, grid=grid(392, 832), stream=stream0)
        del buf37
        del convolution_165
        del primals_497
        del squeeze_496
        del unsqueeze_730
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf43 = aten.convolution_backward(buf42, relu_160, primals_496, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_496
        buf44 = buf43[0]
        buf45 = buf43[1]
        del buf43
        buf46 = buf5; del buf5  # reuse
        buf48 = buf3; del buf3  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_native_batch_norm_backward_threshold_backward_20.run(relu_160, le, buf0, buf44, convolution_164, unsqueeze_742, buf46, buf48, 8192, 98, grid=grid(8192), stream=stream0)
        buf47 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_2.run(buf46, buf47, 2048, 4, grid=grid(2048), stream=stream0)
        buf49 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf51 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_3.run(buf48, squeeze_493, buf49, buf51, 2048, 4, grid=grid(2048), stream=stream0)
        buf50 = buf8; del buf8  # reuse
        buf52 = buf50; del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_21.run(buf52, relu_160, le, buf0, buf44, convolution_164, unsqueeze_742, buf49, squeeze_493, buf47, primals_494, 392, 2048, grid=grid(392, 2048), stream=stream0)
        del convolution_164
        del primals_494
        del squeeze_493
        del unsqueeze_742
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf53 = aten.convolution_backward(buf52, cat_31, primals_493, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_31
        del primals_493
        buf54 = buf53[0]
        buf55 = buf53[1]
        del buf53
        buf56 = reinterpret_tensor(buf40, (208, 4), (1, 208), 0); del buf40  # reuse
        buf58 = empty_strided((208, 4), (1, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(le_6, buf54, convolution_163, unsqueeze_754, buf56, buf58, 832, 98, grid=grid(832), stream=stream0)
        buf57 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(buf56, buf57, 208, 4, grid=grid(208), stream=stream0)
        buf59 = empty((208, ), device='cuda', dtype=torch.float32)
        buf60 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf58, squeeze_490, buf59, buf60, 208, 4, grid=grid(208), stream=stream0)
        buf61 = reinterpret_tensor(buf35, (8, 208, 7, 7), (10192, 1, 1456, 208), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8.run(le_6, buf54, convolution_163, unsqueeze_754, buf59, squeeze_490, buf57, primals_491, buf61, 392, 208, grid=grid(392, 208), stream=stream0)
        del convolution_163
        del le_6
        del primals_491
        del squeeze_490
        del unsqueeze_754
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf62 = aten.convolution_backward(buf61, add_901, primals_490, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_901
        del primals_490
        buf63 = buf62[0]
        buf64 = buf62[1]
        del buf62
        buf65 = buf59; del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_9.run(le_7, buf54, buf63, buf65, 208, 392, grid=grid(208), stream=stream0)
        buf66 = buf58; del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_10.run(le_7, buf54, buf63, convolution_162, unsqueeze_766, buf66, 832, 98, grid=grid(832), stream=stream0)
        buf67 = empty((208, ), device='cuda', dtype=torch.float32)
        buf69 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf66, squeeze_487, buf67, buf69, 208, 4, grid=grid(208), stream=stream0)
        buf68 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_11.run(le_7, buf54, buf63, convolution_162, unsqueeze_766, buf67, squeeze_487, buf65, primals_488, buf68, 392, 208, grid=grid(392, 208), stream=stream0)
        del convolution_162
        del le_7
        del primals_488
        del squeeze_487
        del unsqueeze_766
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf70 = aten.convolution_backward(buf68, add_895, primals_487, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_895
        del primals_487
        buf71 = buf70[0]
        buf72 = buf70[1]
        del buf70
        buf73 = buf67; del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_12.run(le_8, buf54, buf71, buf73, 208, 392, grid=grid(208), stream=stream0)
        buf74 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_13.run(le_8, buf54, buf71, convolution_161, unsqueeze_778, buf74, 832, 98, grid=grid(832), stream=stream0)
        buf75 = empty((208, ), device='cuda', dtype=torch.float32)
        buf77 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf74, squeeze_484, buf75, buf77, 208, 4, grid=grid(208), stream=stream0)
        buf76 = buf68; del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_14.run(le_8, buf54, buf71, convolution_161, unsqueeze_778, buf75, squeeze_484, buf73, primals_485, buf76, 392, 208, grid=grid(392, 208), stream=stream0)
        del convolution_161
        del le_8
        del primals_485
        del squeeze_484
        del unsqueeze_778
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf78 = aten.convolution_backward(buf76, getitem_948, primals_484, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf76
        del getitem_948
        del primals_484
        buf79 = buf78[0]
        buf80 = buf78[1]
        del buf78
        buf81 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_15.run(buf81, le_9, buf79, buf71, buf63, 6656, 49, grid=grid(6656, 49), stream=stream0)
        del buf63
        del buf71
        del le_9
        buf82 = reinterpret_tensor(buf74, (832, ), (1, ), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_16.run(buf81, buf82, 832, 392, grid=grid(832), stream=stream0)
        buf83 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_17.run(buf81, convolution_160, unsqueeze_790, buf83, 3328, 98, grid=grid(3328), stream=stream0)
        buf84 = reinterpret_tensor(buf56, (832, ), (1, ), 0); del buf56  # reuse
        buf85 = empty((832, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_18.run(buf83, squeeze_481, buf84, buf85, 832, 4, grid=grid(832), stream=stream0)
        del buf83
        buf86 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_19.run(buf81, convolution_160, unsqueeze_790, buf84, squeeze_481, buf82, primals_482, buf86, 392, 832, grid=grid(392, 832), stream=stream0)
        del buf81
        del convolution_160
        del primals_482
        del squeeze_481
        del unsqueeze_790
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf87 = aten.convolution_backward(buf86, relu_155, primals_481, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_481
        buf88 = buf87[0]
        buf89 = buf87[1]
        del buf87
        buf90 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.threshold_backward]
        triton_poi_fused_add_div_threshold_backward_22.run(relu_155, relu_160, le, buf0, buf44, buf88, buf90, 392, 2048, grid=grid(392, 2048), stream=stream0)
        del buf0
        del le
        del relu_155
        del relu_160
        buf91 = buf48; del buf48  # reuse
        buf93 = buf46; del buf46  # reuse
        buf100 = empty_strided((2048, 4), (1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_23.run(buf90, convolution_159, unsqueeze_802, convolution_158, unsqueeze_814, buf91, buf93, buf100, 8192, 98, grid=grid(8192), stream=stream0)
        buf92 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_2.run(buf91, buf92, 2048, 4, grid=grid(2048), stream=stream0)
        del buf91
        buf94 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf95 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_3.run(buf93, squeeze_478, buf94, buf95, 2048, 4, grid=grid(2048), stream=stream0)
        del buf93
        buf101 = empty((2048, ), device='cuda', dtype=torch.float32)
        buf102 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_3.run(buf100, squeeze_475, buf101, buf102, 2048, 4, grid=grid(2048), stream=stream0)
        del buf100
        buf96 = reinterpret_tensor(buf88, (8, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf88  # reuse
        buf103 = reinterpret_tensor(buf44, (8, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_24.run(buf90, convolution_159, unsqueeze_802, buf94, squeeze_478, buf92, primals_479, convolution_158, unsqueeze_814, buf101, squeeze_475, primals_476, buf96, buf103, 802816, grid=grid(802816), stream=stream0)
        del buf101
        del buf90
        del buf94
        del convolution_158
        del convolution_159
        del primals_476
        del primals_479
        del squeeze_475
        del squeeze_478
        del unsqueeze_802
        del unsqueeze_814
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf97 = aten.convolution_backward(buf96, relu_150, primals_478, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf96
        del primals_478
        buf98 = buf97[0]
        buf99 = buf97[1]
        del buf97
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf104 = aten.convolution_backward(buf103, cat_30, primals_475, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf103
        del cat_30
        del primals_475
        buf105 = buf104[0]
        buf106 = buf104[1]
        del buf104
        buf107 = reinterpret_tensor(buf86, (8, 208, 14, 14), (40768, 196, 14, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_25.run(buf105, buf107, 326144, grid=grid(326144), stream=stream0)
        buf108 = reinterpret_tensor(buf84, (208, 4), (1, 208), 0); del buf84  # reuse
        buf110 = empty_strided((208, 4), (1, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(le_11, buf105, convolution_157, unsqueeze_826, buf108, buf110, 832, 98, grid=grid(832), stream=stream0)
        buf109 = buf75; del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(buf108, buf109, 208, 4, grid=grid(208), stream=stream0)
        buf111 = empty((208, ), device='cuda', dtype=torch.float32)
        buf112 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf110, squeeze_472, buf111, buf112, 208, 4, grid=grid(208), stream=stream0)
        buf113 = reinterpret_tensor(buf79, (8, 208, 7, 7), (10192, 1, 1456, 208), 0); del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8.run(le_11, buf105, convolution_157, unsqueeze_826, buf111, squeeze_472, buf109, primals_473, buf113, 392, 208, grid=grid(392, 208), stream=stream0)
        del convolution_157
        del le_11
        del primals_473
        del squeeze_472
        del unsqueeze_826
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf114 = aten.convolution_backward(buf113, getitem_930, primals_472, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_930
        del primals_472
        buf115 = buf114[0]
        buf116 = buf114[1]
        del buf114
        buf117 = buf110; del buf110  # reuse
        buf119 = buf108; del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_26.run(le_12, buf105, convolution_156, unsqueeze_838, buf117, buf119, 832, 98, grid=grid(832), stream=stream0)
        buf118 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(buf117, buf118, 208, 4, grid=grid(208), stream=stream0)
        buf120 = empty((208, ), device='cuda', dtype=torch.float32)
        buf121 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf119, squeeze_469, buf120, buf121, 208, 4, grid=grid(208), stream=stream0)
        buf122 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27.run(le_12, buf105, convolution_156, unsqueeze_838, buf120, squeeze_469, buf118, primals_470, buf122, 392, 208, grid=grid(392, 208), stream=stream0)
        del convolution_156
        del le_12
        del primals_470
        del squeeze_469
        del unsqueeze_838
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf123 = aten.convolution_backward(buf122, getitem_923, primals_469, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_923
        del primals_469
        buf124 = buf123[0]
        buf125 = buf123[1]
        del buf123
        buf126 = buf119; del buf119  # reuse
        buf128 = buf117; del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(le_13, buf105, convolution_155, unsqueeze_850, buf126, buf128, 832, 98, grid=grid(832), stream=stream0)
        buf127 = buf120; del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_6.run(buf126, buf127, 208, 4, grid=grid(208), stream=stream0)
        buf129 = empty((208, ), device='cuda', dtype=torch.float32)
        buf130 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_7.run(buf128, squeeze_466, buf129, buf130, 208, 4, grid=grid(208), stream=stream0)
        buf131 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(le_13, buf105, convolution_155, unsqueeze_850, buf129, squeeze_466, buf127, primals_467, buf131, 392, 208, grid=grid(392, 208), stream=stream0)
        del buf105
        del convolution_155
        del le_13
        del primals_467
        del squeeze_466
        del unsqueeze_850
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf132 = aten.convolution_backward(buf131, getitem_916, primals_466, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf131
        del getitem_916
        del primals_466
        buf133 = buf132[0]
        buf134 = buf132[1]
        del buf132
        buf135 = empty((8, 832, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_30.run(le_14, buf133, buf124, buf115, buf107, buf135, 6656, 196, grid=grid(6656, 196), stream=stream0)
        del buf107
        del buf115
        del buf124
        del le_14
        buf136 = reinterpret_tensor(buf128, (832, ), (1, ), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_31.run(buf135, buf136, 832, 1568, grid=grid(832), stream=stream0)
        buf137 = empty((832, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf135, convolution_154, unsqueeze_862, buf137, 10816, 121, grid=grid(10816), stream=stream0)
        buf138 = reinterpret_tensor(buf126, (832, ), (1, ), 0); del buf126  # reuse
        buf139 = empty((832, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_33.run(buf137, squeeze_463, buf138, buf139, 832, 13, grid=grid(832), stream=stream0)
        del buf137
        buf140 = empty_strided((8, 832, 14, 14), (163072, 1, 11648, 832), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_34.run(buf135, convolution_154, unsqueeze_862, buf138, squeeze_463, buf136, primals_464, buf140, 1568, 832, grid=grid(1568, 832), stream=stream0)
        del buf135
        del convolution_154
        del primals_464
        del squeeze_463
        del unsqueeze_862
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf141 = aten.convolution_backward(buf140, relu_150, primals_463, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_463
        buf142 = buf141[0]
        buf143 = buf141[1]
        del buf141
        buf144 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_150, buf98, buf142, buf144, 1024, 1568, grid=grid(1024), stream=stream0)
        buf145 = empty((1024, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_150, buf98, buf142, convolution_153, unsqueeze_874, buf145, 13312, 121, grid=grid(13312), stream=stream0)
        buf146 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf148 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf145, squeeze_460, buf146, buf148, 1024, 13, grid=grid(1024), stream=stream0)
        buf147 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_38.run(relu_150, buf98, buf142, convolution_153, unsqueeze_874, buf146, squeeze_460, buf144, primals_461, buf147, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_153
        del primals_461
        del squeeze_460
        del unsqueeze_874
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf149 = aten.convolution_backward(buf147, cat_29, primals_460, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf147
        del cat_29
        del primals_460
        buf150 = buf149[0]
        buf151 = buf149[1]
        del buf149
        buf152 = empty((104, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_16, buf150, buf152, 1352, 121, grid=grid(1352), stream=stream0)
        buf153 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf152, buf153, 104, 13, grid=grid(104), stream=stream0)
        buf154 = reinterpret_tensor(buf152, (104, 13), (1, 104), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_16, buf150, convolution_152, unsqueeze_886, buf154, 1352, 121, grid=grid(1352), stream=stream0)
        buf155 = empty((104, ), device='cuda', dtype=torch.float32)
        buf156 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf154, squeeze_457, buf155, buf156, 104, 13, grid=grid(104), stream=stream0)
        buf157 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_16, buf150, convolution_152, unsqueeze_886, buf155, squeeze_457, buf153, primals_458, buf157, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_152
        del le_16
        del primals_458
        del squeeze_457
        del unsqueeze_886
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf158 = aten.convolution_backward(buf157, add_842, primals_457, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_842
        del primals_457
        buf159 = buf158[0]
        buf160 = buf158[1]
        del buf158
        buf161 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_17, buf150, buf159, buf161, 104, 1568, grid=grid(104), stream=stream0)
        buf162 = reinterpret_tensor(buf154, (104, 13), (13, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_17, buf150, buf159, convolution_151, unsqueeze_898, buf162, 1352, 121, grid=grid(1352), stream=stream0)
        buf163 = empty((104, ), device='cuda', dtype=torch.float32)
        buf165 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf162, squeeze_454, buf163, buf165, 104, 13, grid=grid(104), stream=stream0)
        buf164 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_17, buf150, buf159, convolution_151, unsqueeze_898, buf163, squeeze_454, buf161, primals_455, buf164, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_151
        del le_17
        del primals_455
        del squeeze_454
        del unsqueeze_898
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf166 = aten.convolution_backward(buf164, add_836, primals_454, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_836
        del primals_454
        buf167 = buf166[0]
        buf168 = buf166[1]
        del buf166
        buf169 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_18, buf150, buf167, buf169, 104, 1568, grid=grid(104), stream=stream0)
        buf170 = buf162; del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_18, buf150, buf167, convolution_150, unsqueeze_910, buf170, 1352, 121, grid=grid(1352), stream=stream0)
        buf171 = empty((104, ), device='cuda', dtype=torch.float32)
        buf173 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf170, squeeze_451, buf171, buf173, 104, 13, grid=grid(104), stream=stream0)
        buf172 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_18, buf150, buf167, convolution_150, unsqueeze_910, buf171, squeeze_451, buf169, primals_452, buf172, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_150
        del le_18
        del primals_452
        del squeeze_451
        del unsqueeze_910
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf174 = aten.convolution_backward(buf172, getitem_886, primals_451, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf172
        del getitem_886
        del primals_451
        buf175 = buf174[0]
        buf176 = buf174[1]
        del buf174
        buf177 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf177, le_19, buf175, buf167, buf159, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf159
        del buf167
        del le_19
        buf178 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf177, buf178, 416, 1568, grid=grid(416), stream=stream0)
        buf179 = empty((416, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf177, convolution_149, unsqueeze_922, buf179, 5408, 121, grid=grid(5408), stream=stream0)
        buf180 = empty((416, ), device='cuda', dtype=torch.float32)
        buf181 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf179, squeeze_448, buf180, buf181, 416, 13, grid=grid(416), stream=stream0)
        buf182 = empty_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf177, convolution_149, unsqueeze_922, buf180, squeeze_448, buf178, primals_449, buf182, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf177
        del convolution_149
        del primals_449
        del squeeze_448
        del unsqueeze_922
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf183 = aten.convolution_backward(buf182, relu_145, primals_448, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_448
        buf184 = buf183[0]
        buf185 = buf183[1]
        del buf183
        buf186 = buf142; del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_56.run(buf186, relu_145, relu_150, buf98, buf184, 8192, 196, grid=grid(8192, 196), stream=stream0)
        del buf184
        del relu_145
        del relu_150
        buf187 = buf146; del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_57.run(buf186, buf187, 1024, 1568, grid=grid(1024), stream=stream0)
        buf188 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_58.run(buf186, convolution_148, unsqueeze_934, buf188, 13312, 121, grid=grid(13312), stream=stream0)
        buf189 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf190 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf188, squeeze_445, buf189, buf190, 1024, 13, grid=grid(1024), stream=stream0)
        buf191 = reinterpret_tensor(buf98, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_59.run(buf186, convolution_148, unsqueeze_934, buf189, squeeze_445, buf187, primals_446, buf191, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_148
        del primals_446
        del squeeze_445
        del unsqueeze_934
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf192 = aten.convolution_backward(buf191, cat_28, primals_445, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_28
        del primals_445
        buf193 = buf192[0]
        buf194 = buf192[1]
        del buf192
        buf195 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_21, buf193, buf195, 1352, 121, grid=grid(1352), stream=stream0)
        buf196 = buf171; del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf195, buf196, 104, 13, grid=grid(104), stream=stream0)
        buf197 = reinterpret_tensor(buf195, (104, 13), (1, 104), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_21, buf193, convolution_147, unsqueeze_946, buf197, 1352, 121, grid=grid(1352), stream=stream0)
        buf198 = empty((104, ), device='cuda', dtype=torch.float32)
        buf199 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf197, squeeze_442, buf198, buf199, 104, 13, grid=grid(104), stream=stream0)
        buf200 = reinterpret_tensor(buf175, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_21, buf193, convolution_147, unsqueeze_946, buf198, squeeze_442, buf196, primals_443, buf200, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_147
        del le_21
        del primals_443
        del squeeze_442
        del unsqueeze_946
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf201 = aten.convolution_backward(buf200, add_814, primals_442, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_814
        del primals_442
        buf202 = buf201[0]
        buf203 = buf201[1]
        del buf201
        buf204 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_22, buf193, buf202, buf204, 104, 1568, grid=grid(104), stream=stream0)
        buf205 = reinterpret_tensor(buf197, (104, 13), (13, 1), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_22, buf193, buf202, convolution_146, unsqueeze_958, buf205, 1352, 121, grid=grid(1352), stream=stream0)
        buf206 = empty((104, ), device='cuda', dtype=torch.float32)
        buf208 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf205, squeeze_439, buf206, buf208, 104, 13, grid=grid(104), stream=stream0)
        buf207 = buf200; del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_22, buf193, buf202, convolution_146, unsqueeze_958, buf206, squeeze_439, buf204, primals_440, buf207, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_146
        del le_22
        del primals_440
        del squeeze_439
        del unsqueeze_958
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf209 = aten.convolution_backward(buf207, add_808, primals_439, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_808
        del primals_439
        buf210 = buf209[0]
        buf211 = buf209[1]
        del buf209
        buf212 = buf206; del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_23, buf193, buf210, buf212, 104, 1568, grid=grid(104), stream=stream0)
        buf213 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_23, buf193, buf210, convolution_145, unsqueeze_970, buf213, 1352, 121, grid=grid(1352), stream=stream0)
        buf214 = empty((104, ), device='cuda', dtype=torch.float32)
        buf216 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf213, squeeze_436, buf214, buf216, 104, 13, grid=grid(104), stream=stream0)
        buf215 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_23, buf193, buf210, convolution_145, unsqueeze_970, buf214, squeeze_436, buf212, primals_437, buf215, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_145
        del le_23
        del primals_437
        del squeeze_436
        del unsqueeze_970
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf217 = aten.convolution_backward(buf215, getitem_856, primals_436, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf215
        del getitem_856
        del primals_436
        buf218 = buf217[0]
        buf219 = buf217[1]
        del buf217
        buf220 = buf193; del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf220, le_24, buf218, buf210, buf202, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf202
        del buf210
        del le_24
        buf221 = buf180; del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf220, buf221, 416, 1568, grid=grid(416), stream=stream0)
        buf222 = buf179; del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf220, convolution_144, unsqueeze_982, buf222, 5408, 121, grid=grid(5408), stream=stream0)
        buf223 = empty((416, ), device='cuda', dtype=torch.float32)
        buf224 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf222, squeeze_433, buf223, buf224, 416, 13, grid=grid(416), stream=stream0)
        buf225 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf220, convolution_144, unsqueeze_982, buf223, squeeze_433, buf221, primals_434, buf225, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf220
        del convolution_144
        del primals_434
        del squeeze_433
        del unsqueeze_982
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf226 = aten.convolution_backward(buf225, relu_140, primals_433, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_433
        buf227 = buf226[0]
        buf228 = buf226[1]
        del buf226
        buf229 = buf189; del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_140, buf186, buf227, buf229, 1024, 1568, grid=grid(1024), stream=stream0)
        buf230 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_140, buf186, buf227, convolution_143, unsqueeze_994, buf230, 13312, 121, grid=grid(13312), stream=stream0)
        buf231 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf233 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf230, squeeze_430, buf231, buf233, 1024, 13, grid=grid(1024), stream=stream0)
        buf232 = buf191; del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_38.run(relu_140, buf186, buf227, convolution_143, unsqueeze_994, buf231, squeeze_430, buf229, primals_431, buf232, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_143
        del primals_431
        del squeeze_430
        del unsqueeze_994
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf234 = aten.convolution_backward(buf232, cat_27, primals_430, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf232
        del cat_27
        del primals_430
        buf235 = buf234[0]
        buf236 = buf234[1]
        del buf234
        buf237 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_26, buf235, buf237, 1352, 121, grid=grid(1352), stream=stream0)
        buf238 = buf214; del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf237, buf238, 104, 13, grid=grid(104), stream=stream0)
        buf239 = reinterpret_tensor(buf237, (104, 13), (1, 104), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_26, buf235, convolution_142, unsqueeze_1006, buf239, 1352, 121, grid=grid(1352), stream=stream0)
        buf240 = empty((104, ), device='cuda', dtype=torch.float32)
        buf241 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf239, squeeze_427, buf240, buf241, 104, 13, grid=grid(104), stream=stream0)
        buf242 = reinterpret_tensor(buf218, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_26, buf235, convolution_142, unsqueeze_1006, buf240, squeeze_427, buf238, primals_428, buf242, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_142
        del le_26
        del primals_428
        del squeeze_427
        del unsqueeze_1006
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf243 = aten.convolution_backward(buf242, add_786, primals_427, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_786
        del primals_427
        buf244 = buf243[0]
        buf245 = buf243[1]
        del buf243
        buf246 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_27, buf235, buf244, buf246, 104, 1568, grid=grid(104), stream=stream0)
        buf247 = reinterpret_tensor(buf239, (104, 13), (13, 1), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_27, buf235, buf244, convolution_141, unsqueeze_1018, buf247, 1352, 121, grid=grid(1352), stream=stream0)
        buf248 = empty((104, ), device='cuda', dtype=torch.float32)
        buf250 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf247, squeeze_424, buf248, buf250, 104, 13, grid=grid(104), stream=stream0)
        buf249 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_27, buf235, buf244, convolution_141, unsqueeze_1018, buf248, squeeze_424, buf246, primals_425, buf249, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_141
        del le_27
        del primals_425
        del squeeze_424
        del unsqueeze_1018
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf251 = aten.convolution_backward(buf249, add_780, primals_424, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_780
        del primals_424
        buf252 = buf251[0]
        buf253 = buf251[1]
        del buf251
        buf254 = buf248; del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_28, buf235, buf252, buf254, 104, 1568, grid=grid(104), stream=stream0)
        buf255 = buf247; del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_28, buf235, buf252, convolution_140, unsqueeze_1030, buf255, 1352, 121, grid=grid(1352), stream=stream0)
        buf256 = empty((104, ), device='cuda', dtype=torch.float32)
        buf258 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf255, squeeze_421, buf256, buf258, 104, 13, grid=grid(104), stream=stream0)
        buf257 = buf249; del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_28, buf235, buf252, convolution_140, unsqueeze_1030, buf256, squeeze_421, buf254, primals_422, buf257, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_140
        del le_28
        del primals_422
        del squeeze_421
        del unsqueeze_1030
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf259 = aten.convolution_backward(buf257, getitem_826, primals_421, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf257
        del getitem_826
        del primals_421
        buf260 = buf259[0]
        buf261 = buf259[1]
        del buf259
        buf262 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf262, le_29, buf260, buf252, buf244, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf244
        del buf252
        del le_29
        buf263 = buf223; del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf262, buf263, 416, 1568, grid=grid(416), stream=stream0)
        buf264 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf262, convolution_139, unsqueeze_1042, buf264, 5408, 121, grid=grid(5408), stream=stream0)
        buf265 = empty((416, ), device='cuda', dtype=torch.float32)
        buf266 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf264, squeeze_418, buf265, buf266, 416, 13, grid=grid(416), stream=stream0)
        buf267 = buf225; del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf262, convolution_139, unsqueeze_1042, buf265, squeeze_418, buf263, primals_419, buf267, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf262
        del convolution_139
        del primals_419
        del squeeze_418
        del unsqueeze_1042
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf268 = aten.convolution_backward(buf267, relu_135, primals_418, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_418
        buf269 = buf268[0]
        buf270 = buf268[1]
        del buf268
        buf271 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_60.run(buf271, relu_135, relu_140, buf227, buf269, 8192, 196, grid=grid(8192, 196), stream=stream0)
        del buf227
        del relu_135
        del relu_140
        buf272 = buf231; del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_57.run(buf271, buf272, 1024, 1568, grid=grid(1024), stream=stream0)
        buf273 = buf230; del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_58.run(buf271, convolution_138, unsqueeze_1054, buf273, 13312, 121, grid=grid(13312), stream=stream0)
        buf274 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf275 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf273, squeeze_415, buf274, buf275, 1024, 13, grid=grid(1024), stream=stream0)
        buf276 = reinterpret_tensor(buf269, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_59.run(buf271, convolution_138, unsqueeze_1054, buf274, squeeze_415, buf272, primals_416, buf276, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_138
        del primals_416
        del squeeze_415
        del unsqueeze_1054
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf277 = aten.convolution_backward(buf276, cat_26, primals_415, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_26
        del primals_415
        buf278 = buf277[0]
        buf279 = buf277[1]
        del buf277
        buf280 = buf255; del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_31, buf278, buf280, 1352, 121, grid=grid(1352), stream=stream0)
        buf281 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf280, buf281, 104, 13, grid=grid(104), stream=stream0)
        buf282 = reinterpret_tensor(buf280, (104, 13), (1, 104), 0); del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_31, buf278, convolution_137, unsqueeze_1066, buf282, 1352, 121, grid=grid(1352), stream=stream0)
        buf283 = empty((104, ), device='cuda', dtype=torch.float32)
        buf284 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf282, squeeze_412, buf283, buf284, 104, 13, grid=grid(104), stream=stream0)
        buf285 = reinterpret_tensor(buf260, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_31, buf278, convolution_137, unsqueeze_1066, buf283, squeeze_412, buf281, primals_413, buf285, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_137
        del le_31
        del primals_413
        del squeeze_412
        del unsqueeze_1066
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf286 = aten.convolution_backward(buf285, add_758, primals_412, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_758
        del primals_412
        buf287 = buf286[0]
        buf288 = buf286[1]
        del buf286
        buf289 = buf283; del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_32, buf278, buf287, buf289, 104, 1568, grid=grid(104), stream=stream0)
        buf290 = reinterpret_tensor(buf282, (104, 13), (13, 1), 0); del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_32, buf278, buf287, convolution_136, unsqueeze_1078, buf290, 1352, 121, grid=grid(1352), stream=stream0)
        buf291 = empty((104, ), device='cuda', dtype=torch.float32)
        buf293 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf290, squeeze_409, buf291, buf293, 104, 13, grid=grid(104), stream=stream0)
        buf292 = buf285; del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_32, buf278, buf287, convolution_136, unsqueeze_1078, buf291, squeeze_409, buf289, primals_410, buf292, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_136
        del le_32
        del primals_410
        del squeeze_409
        del unsqueeze_1078
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf294 = aten.convolution_backward(buf292, add_752, primals_409, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_752
        del primals_409
        buf295 = buf294[0]
        buf296 = buf294[1]
        del buf294
        buf297 = buf291; del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_33, buf278, buf295, buf297, 104, 1568, grid=grid(104), stream=stream0)
        buf298 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_33, buf278, buf295, convolution_135, unsqueeze_1090, buf298, 1352, 121, grid=grid(1352), stream=stream0)
        buf299 = empty((104, ), device='cuda', dtype=torch.float32)
        buf301 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf298, squeeze_406, buf299, buf301, 104, 13, grid=grid(104), stream=stream0)
        buf300 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_33, buf278, buf295, convolution_135, unsqueeze_1090, buf299, squeeze_406, buf297, primals_407, buf300, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_135
        del le_33
        del primals_407
        del squeeze_406
        del unsqueeze_1090
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf302 = aten.convolution_backward(buf300, getitem_796, primals_406, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf300
        del getitem_796
        del primals_406
        buf303 = buf302[0]
        buf304 = buf302[1]
        del buf302
        buf305 = buf278; del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf305, le_34, buf303, buf295, buf287, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf287
        del buf295
        del le_34
        buf306 = buf265; del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf305, buf306, 416, 1568, grid=grid(416), stream=stream0)
        buf307 = buf264; del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf305, convolution_134, unsqueeze_1102, buf307, 5408, 121, grid=grid(5408), stream=stream0)
        buf308 = empty((416, ), device='cuda', dtype=torch.float32)
        buf309 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf307, squeeze_403, buf308, buf309, 416, 13, grid=grid(416), stream=stream0)
        buf310 = buf267; del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf305, convolution_134, unsqueeze_1102, buf308, squeeze_403, buf306, primals_404, buf310, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf305
        del convolution_134
        del primals_404
        del squeeze_403
        del unsqueeze_1102
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf311 = aten.convolution_backward(buf310, relu_130, primals_403, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_403
        buf312 = buf311[0]
        buf313 = buf311[1]
        del buf311
        buf314 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_130, buf271, buf312, buf314, 1024, 1568, grid=grid(1024), stream=stream0)
        buf315 = buf273; del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_130, buf271, buf312, convolution_133, unsqueeze_1114, buf315, 13312, 121, grid=grid(13312), stream=stream0)
        buf316 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf318 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf315, squeeze_400, buf316, buf318, 1024, 13, grid=grid(1024), stream=stream0)
        buf317 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_38.run(relu_130, buf271, buf312, convolution_133, unsqueeze_1114, buf316, squeeze_400, buf314, primals_401, buf317, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_133
        del primals_401
        del squeeze_400
        del unsqueeze_1114
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf319 = aten.convolution_backward(buf317, cat_25, primals_400, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf317
        del cat_25
        del primals_400
        buf320 = buf319[0]
        buf321 = buf319[1]
        del buf319
        buf322 = buf298; del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_36, buf320, buf322, 1352, 121, grid=grid(1352), stream=stream0)
        buf323 = buf299; del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf322, buf323, 104, 13, grid=grid(104), stream=stream0)
        buf324 = reinterpret_tensor(buf322, (104, 13), (1, 104), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_36, buf320, convolution_132, unsqueeze_1126, buf324, 1352, 121, grid=grid(1352), stream=stream0)
        buf325 = empty((104, ), device='cuda', dtype=torch.float32)
        buf326 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf324, squeeze_397, buf325, buf326, 104, 13, grid=grid(104), stream=stream0)
        buf327 = reinterpret_tensor(buf303, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_36, buf320, convolution_132, unsqueeze_1126, buf325, squeeze_397, buf323, primals_398, buf327, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_132
        del le_36
        del primals_398
        del squeeze_397
        del unsqueeze_1126
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf328 = aten.convolution_backward(buf327, add_730, primals_397, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_730
        del primals_397
        buf329 = buf328[0]
        buf330 = buf328[1]
        del buf328
        buf331 = buf325; del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_37, buf320, buf329, buf331, 104, 1568, grid=grid(104), stream=stream0)
        buf332 = reinterpret_tensor(buf324, (104, 13), (13, 1), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_37, buf320, buf329, convolution_131, unsqueeze_1138, buf332, 1352, 121, grid=grid(1352), stream=stream0)
        buf333 = empty((104, ), device='cuda', dtype=torch.float32)
        buf335 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf332, squeeze_394, buf333, buf335, 104, 13, grid=grid(104), stream=stream0)
        buf334 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_37, buf320, buf329, convolution_131, unsqueeze_1138, buf333, squeeze_394, buf331, primals_395, buf334, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_131
        del le_37
        del primals_395
        del squeeze_394
        del unsqueeze_1138
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf336 = aten.convolution_backward(buf334, add_724, primals_394, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_724
        del primals_394
        buf337 = buf336[0]
        buf338 = buf336[1]
        del buf336
        buf339 = buf333; del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_38, buf320, buf337, buf339, 104, 1568, grid=grid(104), stream=stream0)
        buf340 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_38, buf320, buf337, convolution_130, unsqueeze_1150, buf340, 1352, 121, grid=grid(1352), stream=stream0)
        buf341 = empty((104, ), device='cuda', dtype=torch.float32)
        buf343 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf340, squeeze_391, buf341, buf343, 104, 13, grid=grid(104), stream=stream0)
        buf342 = buf334; del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_38, buf320, buf337, convolution_130, unsqueeze_1150, buf341, squeeze_391, buf339, primals_392, buf342, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_130
        del le_38
        del primals_392
        del squeeze_391
        del unsqueeze_1150
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf344 = aten.convolution_backward(buf342, getitem_766, primals_391, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf342
        del getitem_766
        del primals_391
        buf345 = buf344[0]
        buf346 = buf344[1]
        del buf344
        buf347 = buf320; del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf347, le_39, buf345, buf337, buf329, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf329
        del buf337
        del le_39
        buf348 = buf308; del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf347, buf348, 416, 1568, grid=grid(416), stream=stream0)
        buf349 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf347, convolution_129, unsqueeze_1162, buf349, 5408, 121, grid=grid(5408), stream=stream0)
        buf350 = empty((416, ), device='cuda', dtype=torch.float32)
        buf351 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf349, squeeze_388, buf350, buf351, 416, 13, grid=grid(416), stream=stream0)
        buf352 = buf310; del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf347, convolution_129, unsqueeze_1162, buf350, squeeze_388, buf348, primals_389, buf352, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf347
        del convolution_129
        del primals_389
        del squeeze_388
        del unsqueeze_1162
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf353 = aten.convolution_backward(buf352, relu_125, primals_388, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_388
        buf354 = buf353[0]
        buf355 = buf353[1]
        del buf353
        buf356 = buf271; del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_60.run(buf356, relu_125, relu_130, buf312, buf354, 8192, 196, grid=grid(8192, 196), stream=stream0)
        del buf312
        del relu_125
        del relu_130
        buf357 = buf316; del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_57.run(buf356, buf357, 1024, 1568, grid=grid(1024), stream=stream0)
        buf358 = buf315; del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_58.run(buf356, convolution_128, unsqueeze_1174, buf358, 13312, 121, grid=grid(13312), stream=stream0)
        buf359 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf360 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf358, squeeze_385, buf359, buf360, 1024, 13, grid=grid(1024), stream=stream0)
        buf361 = reinterpret_tensor(buf354, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_59.run(buf356, convolution_128, unsqueeze_1174, buf359, squeeze_385, buf357, primals_386, buf361, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_128
        del primals_386
        del squeeze_385
        del unsqueeze_1174
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf362 = aten.convolution_backward(buf361, cat_24, primals_385, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_24
        del primals_385
        buf363 = buf362[0]
        buf364 = buf362[1]
        del buf362
        buf365 = buf340; del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_41, buf363, buf365, 1352, 121, grid=grid(1352), stream=stream0)
        buf366 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf365, buf366, 104, 13, grid=grid(104), stream=stream0)
        buf367 = reinterpret_tensor(buf365, (104, 13), (1, 104), 0); del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_41, buf363, convolution_127, unsqueeze_1186, buf367, 1352, 121, grid=grid(1352), stream=stream0)
        buf368 = empty((104, ), device='cuda', dtype=torch.float32)
        buf369 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf367, squeeze_382, buf368, buf369, 104, 13, grid=grid(104), stream=stream0)
        buf370 = reinterpret_tensor(buf345, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_41, buf363, convolution_127, unsqueeze_1186, buf368, squeeze_382, buf366, primals_383, buf370, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_127
        del le_41
        del primals_383
        del squeeze_382
        del unsqueeze_1186
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf371 = aten.convolution_backward(buf370, add_702, primals_382, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_702
        del primals_382
        buf372 = buf371[0]
        buf373 = buf371[1]
        del buf371
        buf374 = buf368; del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_42, buf363, buf372, buf374, 104, 1568, grid=grid(104), stream=stream0)
        buf375 = reinterpret_tensor(buf367, (104, 13), (13, 1), 0); del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_42, buf363, buf372, convolution_126, unsqueeze_1198, buf375, 1352, 121, grid=grid(1352), stream=stream0)
        buf376 = empty((104, ), device='cuda', dtype=torch.float32)
        buf378 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf375, squeeze_379, buf376, buf378, 104, 13, grid=grid(104), stream=stream0)
        buf377 = buf370; del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_42, buf363, buf372, convolution_126, unsqueeze_1198, buf376, squeeze_379, buf374, primals_380, buf377, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_126
        del le_42
        del primals_380
        del squeeze_379
        del unsqueeze_1198
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf379 = aten.convolution_backward(buf377, add_696, primals_379, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_696
        del primals_379
        buf380 = buf379[0]
        buf381 = buf379[1]
        del buf379
        buf382 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_43, buf363, buf380, buf382, 104, 1568, grid=grid(104), stream=stream0)
        buf383 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_43, buf363, buf380, convolution_125, unsqueeze_1210, buf383, 1352, 121, grid=grid(1352), stream=stream0)
        buf384 = empty((104, ), device='cuda', dtype=torch.float32)
        buf386 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf383, squeeze_376, buf384, buf386, 104, 13, grid=grid(104), stream=stream0)
        buf385 = buf377; del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_43, buf363, buf380, convolution_125, unsqueeze_1210, buf384, squeeze_376, buf382, primals_377, buf385, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_125
        del le_43
        del primals_377
        del squeeze_376
        del unsqueeze_1210
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf387 = aten.convolution_backward(buf385, getitem_736, primals_376, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf385
        del getitem_736
        del primals_376
        buf388 = buf387[0]
        buf389 = buf387[1]
        del buf387
        buf390 = buf363; del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf390, le_44, buf388, buf380, buf372, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf372
        del buf380
        del le_44
        buf391 = buf350; del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf390, buf391, 416, 1568, grid=grid(416), stream=stream0)
        buf392 = buf349; del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf390, convolution_124, unsqueeze_1222, buf392, 5408, 121, grid=grid(5408), stream=stream0)
        buf393 = empty((416, ), device='cuda', dtype=torch.float32)
        buf394 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf392, squeeze_373, buf393, buf394, 416, 13, grid=grid(416), stream=stream0)
        buf395 = buf352; del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf390, convolution_124, unsqueeze_1222, buf393, squeeze_373, buf391, primals_374, buf395, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf390
        del convolution_124
        del primals_374
        del squeeze_373
        del unsqueeze_1222
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf396 = aten.convolution_backward(buf395, relu_120, primals_373, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_373
        buf397 = buf396[0]
        buf398 = buf396[1]
        del buf396
        buf399 = buf359; del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_120, buf356, buf397, buf399, 1024, 1568, grid=grid(1024), stream=stream0)
        buf400 = buf358; del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_120, buf356, buf397, convolution_123, unsqueeze_1234, buf400, 13312, 121, grid=grid(13312), stream=stream0)
        buf401 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf403 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf400, squeeze_370, buf401, buf403, 1024, 13, grid=grid(1024), stream=stream0)
        buf402 = buf361; del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_38.run(relu_120, buf356, buf397, convolution_123, unsqueeze_1234, buf401, squeeze_370, buf399, primals_371, buf402, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_123
        del primals_371
        del squeeze_370
        del unsqueeze_1234
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf404 = aten.convolution_backward(buf402, cat_23, primals_370, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf402
        del cat_23
        del primals_370
        buf405 = buf404[0]
        buf406 = buf404[1]
        del buf404
        buf407 = buf383; del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_46, buf405, buf407, 1352, 121, grid=grid(1352), stream=stream0)
        buf408 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf407, buf408, 104, 13, grid=grid(104), stream=stream0)
        buf409 = reinterpret_tensor(buf407, (104, 13), (1, 104), 0); del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_46, buf405, convolution_122, unsqueeze_1246, buf409, 1352, 121, grid=grid(1352), stream=stream0)
        buf410 = empty((104, ), device='cuda', dtype=torch.float32)
        buf411 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf409, squeeze_367, buf410, buf411, 104, 13, grid=grid(104), stream=stream0)
        buf412 = reinterpret_tensor(buf388, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_46, buf405, convolution_122, unsqueeze_1246, buf410, squeeze_367, buf408, primals_368, buf412, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_122
        del le_46
        del primals_368
        del squeeze_367
        del unsqueeze_1246
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf413 = aten.convolution_backward(buf412, add_674, primals_367, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_674
        del primals_367
        buf414 = buf413[0]
        buf415 = buf413[1]
        del buf413
        buf416 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_47, buf405, buf414, buf416, 104, 1568, grid=grid(104), stream=stream0)
        buf417 = reinterpret_tensor(buf409, (104, 13), (13, 1), 0); del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_47, buf405, buf414, convolution_121, unsqueeze_1258, buf417, 1352, 121, grid=grid(1352), stream=stream0)
        buf418 = empty((104, ), device='cuda', dtype=torch.float32)
        buf420 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf417, squeeze_364, buf418, buf420, 104, 13, grid=grid(104), stream=stream0)
        buf419 = buf412; del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_47, buf405, buf414, convolution_121, unsqueeze_1258, buf418, squeeze_364, buf416, primals_365, buf419, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_121
        del le_47
        del primals_365
        del squeeze_364
        del unsqueeze_1258
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf421 = aten.convolution_backward(buf419, add_668, primals_364, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_668
        del primals_364
        buf422 = buf421[0]
        buf423 = buf421[1]
        del buf421
        buf424 = buf418; del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_48, buf405, buf422, buf424, 104, 1568, grid=grid(104), stream=stream0)
        buf425 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_48, buf405, buf422, convolution_120, unsqueeze_1270, buf425, 1352, 121, grid=grid(1352), stream=stream0)
        buf426 = empty((104, ), device='cuda', dtype=torch.float32)
        buf428 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf425, squeeze_361, buf426, buf428, 104, 13, grid=grid(104), stream=stream0)
        buf427 = buf419; del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_48, buf405, buf422, convolution_120, unsqueeze_1270, buf426, squeeze_361, buf424, primals_362, buf427, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_120
        del le_48
        del primals_362
        del squeeze_361
        del unsqueeze_1270
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf429 = aten.convolution_backward(buf427, getitem_706, primals_361, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf427
        del getitem_706
        del primals_361
        buf430 = buf429[0]
        buf431 = buf429[1]
        del buf429
        buf432 = buf405; del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf432, le_49, buf430, buf422, buf414, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf414
        del buf422
        del le_49
        buf433 = buf393; del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf432, buf433, 416, 1568, grid=grid(416), stream=stream0)
        buf434 = buf392; del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf432, convolution_119, unsqueeze_1282, buf434, 5408, 121, grid=grid(5408), stream=stream0)
        buf435 = empty((416, ), device='cuda', dtype=torch.float32)
        buf436 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf434, squeeze_358, buf435, buf436, 416, 13, grid=grid(416), stream=stream0)
        buf437 = buf395; del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf432, convolution_119, unsqueeze_1282, buf435, squeeze_358, buf433, primals_359, buf437, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf432
        del convolution_119
        del primals_359
        del squeeze_358
        del unsqueeze_1282
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf438 = aten.convolution_backward(buf437, relu_115, primals_358, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_358
        buf439 = buf438[0]
        buf440 = buf438[1]
        del buf438
        buf441 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_60.run(buf441, relu_115, relu_120, buf397, buf439, 8192, 196, grid=grid(8192, 196), stream=stream0)
        del buf397
        del relu_115
        del relu_120
        buf442 = buf401; del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_57.run(buf441, buf442, 1024, 1568, grid=grid(1024), stream=stream0)
        buf443 = buf400; del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_58.run(buf441, convolution_118, unsqueeze_1294, buf443, 13312, 121, grid=grid(13312), stream=stream0)
        buf444 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf445 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf443, squeeze_355, buf444, buf445, 1024, 13, grid=grid(1024), stream=stream0)
        buf446 = reinterpret_tensor(buf439, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf439  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_59.run(buf441, convolution_118, unsqueeze_1294, buf444, squeeze_355, buf442, primals_356, buf446, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_118
        del primals_356
        del squeeze_355
        del unsqueeze_1294
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf447 = aten.convolution_backward(buf446, cat_22, primals_355, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_22
        del primals_355
        buf448 = buf447[0]
        buf449 = buf447[1]
        del buf447
        buf450 = buf425; del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_51, buf448, buf450, 1352, 121, grid=grid(1352), stream=stream0)
        buf451 = buf426; del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf450, buf451, 104, 13, grid=grid(104), stream=stream0)
        buf452 = reinterpret_tensor(buf450, (104, 13), (1, 104), 0); del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_51, buf448, convolution_117, unsqueeze_1306, buf452, 1352, 121, grid=grid(1352), stream=stream0)
        buf453 = empty((104, ), device='cuda', dtype=torch.float32)
        buf454 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf452, squeeze_352, buf453, buf454, 104, 13, grid=grid(104), stream=stream0)
        buf455 = reinterpret_tensor(buf430, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_51, buf448, convolution_117, unsqueeze_1306, buf453, squeeze_352, buf451, primals_353, buf455, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_117
        del le_51
        del primals_353
        del squeeze_352
        del unsqueeze_1306
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf456 = aten.convolution_backward(buf455, add_646, primals_352, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_646
        del primals_352
        buf457 = buf456[0]
        buf458 = buf456[1]
        del buf456
        buf459 = buf453; del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_52, buf448, buf457, buf459, 104, 1568, grid=grid(104), stream=stream0)
        buf460 = reinterpret_tensor(buf452, (104, 13), (13, 1), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_52, buf448, buf457, convolution_116, unsqueeze_1318, buf460, 1352, 121, grid=grid(1352), stream=stream0)
        buf461 = empty((104, ), device='cuda', dtype=torch.float32)
        buf463 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf460, squeeze_349, buf461, buf463, 104, 13, grid=grid(104), stream=stream0)
        buf462 = buf455; del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_52, buf448, buf457, convolution_116, unsqueeze_1318, buf461, squeeze_349, buf459, primals_350, buf462, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_116
        del le_52
        del primals_350
        del squeeze_349
        del unsqueeze_1318
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf464 = aten.convolution_backward(buf462, add_640, primals_349, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_640
        del primals_349
        buf465 = buf464[0]
        buf466 = buf464[1]
        del buf464
        buf467 = buf461; del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_53, buf448, buf465, buf467, 104, 1568, grid=grid(104), stream=stream0)
        buf468 = buf460; del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_53, buf448, buf465, convolution_115, unsqueeze_1330, buf468, 1352, 121, grid=grid(1352), stream=stream0)
        buf469 = empty((104, ), device='cuda', dtype=torch.float32)
        buf471 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf468, squeeze_346, buf469, buf471, 104, 13, grid=grid(104), stream=stream0)
        buf470 = buf462; del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_53, buf448, buf465, convolution_115, unsqueeze_1330, buf469, squeeze_346, buf467, primals_347, buf470, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_115
        del le_53
        del primals_347
        del squeeze_346
        del unsqueeze_1330
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf472 = aten.convolution_backward(buf470, getitem_676, primals_346, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf470
        del getitem_676
        del primals_346
        buf473 = buf472[0]
        buf474 = buf472[1]
        del buf472
        buf475 = buf448; del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf475, le_54, buf473, buf465, buf457, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf457
        del buf465
        del le_54
        buf476 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf475, buf476, 416, 1568, grid=grid(416), stream=stream0)
        buf477 = buf434; del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf475, convolution_114, unsqueeze_1342, buf477, 5408, 121, grid=grid(5408), stream=stream0)
        buf478 = empty((416, ), device='cuda', dtype=torch.float32)
        buf479 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf477, squeeze_343, buf478, buf479, 416, 13, grid=grid(416), stream=stream0)
        buf480 = buf437; del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf475, convolution_114, unsqueeze_1342, buf478, squeeze_343, buf476, primals_344, buf480, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf475
        del convolution_114
        del primals_344
        del squeeze_343
        del unsqueeze_1342
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf481 = aten.convolution_backward(buf480, relu_110, primals_343, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_343
        buf482 = buf481[0]
        buf483 = buf481[1]
        del buf481
        buf484 = buf444; del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_110, buf441, buf482, buf484, 1024, 1568, grid=grid(1024), stream=stream0)
        buf485 = buf443; del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_110, buf441, buf482, convolution_113, unsqueeze_1354, buf485, 13312, 121, grid=grid(13312), stream=stream0)
        buf486 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf488 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf485, squeeze_340, buf486, buf488, 1024, 13, grid=grid(1024), stream=stream0)
        buf487 = buf446; del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_38.run(relu_110, buf441, buf482, convolution_113, unsqueeze_1354, buf486, squeeze_340, buf484, primals_341, buf487, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_113
        del primals_341
        del squeeze_340
        del unsqueeze_1354
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf489 = aten.convolution_backward(buf487, cat_21, primals_340, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf487
        del cat_21
        del primals_340
        buf490 = buf489[0]
        buf491 = buf489[1]
        del buf489
        buf492 = buf468; del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_56, buf490, buf492, 1352, 121, grid=grid(1352), stream=stream0)
        buf493 = buf469; del buf469  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf492, buf493, 104, 13, grid=grid(104), stream=stream0)
        buf494 = reinterpret_tensor(buf492, (104, 13), (1, 104), 0); del buf492  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_56, buf490, convolution_112, unsqueeze_1366, buf494, 1352, 121, grid=grid(1352), stream=stream0)
        buf495 = empty((104, ), device='cuda', dtype=torch.float32)
        buf496 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf494, squeeze_337, buf495, buf496, 104, 13, grid=grid(104), stream=stream0)
        buf497 = reinterpret_tensor(buf473, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_56, buf490, convolution_112, unsqueeze_1366, buf495, squeeze_337, buf493, primals_338, buf497, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_112
        del le_56
        del primals_338
        del squeeze_337
        del unsqueeze_1366
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf498 = aten.convolution_backward(buf497, add_618, primals_337, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_618
        del primals_337
        buf499 = buf498[0]
        buf500 = buf498[1]
        del buf498
        buf501 = buf495; del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_57, buf490, buf499, buf501, 104, 1568, grid=grid(104), stream=stream0)
        buf502 = reinterpret_tensor(buf494, (104, 13), (13, 1), 0); del buf494  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_57, buf490, buf499, convolution_111, unsqueeze_1378, buf502, 1352, 121, grid=grid(1352), stream=stream0)
        buf503 = empty((104, ), device='cuda', dtype=torch.float32)
        buf505 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf502, squeeze_334, buf503, buf505, 104, 13, grid=grid(104), stream=stream0)
        buf504 = buf497; del buf497  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_57, buf490, buf499, convolution_111, unsqueeze_1378, buf503, squeeze_334, buf501, primals_335, buf504, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_111
        del le_57
        del primals_335
        del squeeze_334
        del unsqueeze_1378
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf506 = aten.convolution_backward(buf504, add_612, primals_334, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_612
        del primals_334
        buf507 = buf506[0]
        buf508 = buf506[1]
        del buf506
        buf509 = buf503; del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_58, buf490, buf507, buf509, 104, 1568, grid=grid(104), stream=stream0)
        buf510 = buf502; del buf502  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_58, buf490, buf507, convolution_110, unsqueeze_1390, buf510, 1352, 121, grid=grid(1352), stream=stream0)
        buf511 = empty((104, ), device='cuda', dtype=torch.float32)
        buf513 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf510, squeeze_331, buf511, buf513, 104, 13, grid=grid(104), stream=stream0)
        buf512 = buf504; del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_58, buf490, buf507, convolution_110, unsqueeze_1390, buf511, squeeze_331, buf509, primals_332, buf512, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_110
        del le_58
        del primals_332
        del squeeze_331
        del unsqueeze_1390
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf514 = aten.convolution_backward(buf512, getitem_646, primals_331, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf512
        del getitem_646
        del primals_331
        buf515 = buf514[0]
        buf516 = buf514[1]
        del buf514
        buf517 = buf490; del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf517, le_59, buf515, buf507, buf499, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf499
        del buf507
        del le_59
        buf518 = buf478; del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf517, buf518, 416, 1568, grid=grid(416), stream=stream0)
        buf519 = buf477; del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf517, convolution_109, unsqueeze_1402, buf519, 5408, 121, grid=grid(5408), stream=stream0)
        buf520 = empty((416, ), device='cuda', dtype=torch.float32)
        buf521 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf519, squeeze_328, buf520, buf521, 416, 13, grid=grid(416), stream=stream0)
        buf522 = buf480; del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf517, convolution_109, unsqueeze_1402, buf520, squeeze_328, buf518, primals_329, buf522, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf517
        del convolution_109
        del primals_329
        del squeeze_328
        del unsqueeze_1402
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf523 = aten.convolution_backward(buf522, relu_105, primals_328, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_328
        buf524 = buf523[0]
        buf525 = buf523[1]
        del buf523
        buf526 = buf441; del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_60.run(buf526, relu_105, relu_110, buf482, buf524, 8192, 196, grid=grid(8192, 196), stream=stream0)
        del buf482
        del relu_105
        del relu_110
        buf527 = buf486; del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_57.run(buf526, buf527, 1024, 1568, grid=grid(1024), stream=stream0)
        buf528 = buf485; del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_58.run(buf526, convolution_108, unsqueeze_1414, buf528, 13312, 121, grid=grid(13312), stream=stream0)
        buf529 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf530 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf528, squeeze_325, buf529, buf530, 1024, 13, grid=grid(1024), stream=stream0)
        buf531 = reinterpret_tensor(buf524, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf524  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_59.run(buf526, convolution_108, unsqueeze_1414, buf529, squeeze_325, buf527, primals_326, buf531, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_108
        del primals_326
        del squeeze_325
        del unsqueeze_1414
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf532 = aten.convolution_backward(buf531, cat_20, primals_325, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_20
        del primals_325
        buf533 = buf532[0]
        buf534 = buf532[1]
        del buf532
        buf535 = buf510; del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_61, buf533, buf535, 1352, 121, grid=grid(1352), stream=stream0)
        buf536 = buf511; del buf511  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf535, buf536, 104, 13, grid=grid(104), stream=stream0)
        buf537 = reinterpret_tensor(buf535, (104, 13), (1, 104), 0); del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_61, buf533, convolution_107, unsqueeze_1426, buf537, 1352, 121, grid=grid(1352), stream=stream0)
        buf538 = empty((104, ), device='cuda', dtype=torch.float32)
        buf539 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf537, squeeze_322, buf538, buf539, 104, 13, grid=grid(104), stream=stream0)
        buf540 = reinterpret_tensor(buf515, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf515  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_61, buf533, convolution_107, unsqueeze_1426, buf538, squeeze_322, buf536, primals_323, buf540, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_107
        del le_61
        del primals_323
        del squeeze_322
        del unsqueeze_1426
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf541 = aten.convolution_backward(buf540, add_590, primals_322, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_590
        del primals_322
        buf542 = buf541[0]
        buf543 = buf541[1]
        del buf541
        buf544 = buf538; del buf538  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_62, buf533, buf542, buf544, 104, 1568, grid=grid(104), stream=stream0)
        buf545 = reinterpret_tensor(buf537, (104, 13), (13, 1), 0); del buf537  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_62, buf533, buf542, convolution_106, unsqueeze_1438, buf545, 1352, 121, grid=grid(1352), stream=stream0)
        buf546 = empty((104, ), device='cuda', dtype=torch.float32)
        buf548 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf545, squeeze_319, buf546, buf548, 104, 13, grid=grid(104), stream=stream0)
        buf547 = buf540; del buf540  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_62, buf533, buf542, convolution_106, unsqueeze_1438, buf546, squeeze_319, buf544, primals_320, buf547, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_106
        del le_62
        del primals_320
        del squeeze_319
        del unsqueeze_1438
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf549 = aten.convolution_backward(buf547, add_584, primals_319, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_584
        del primals_319
        buf550 = buf549[0]
        buf551 = buf549[1]
        del buf549
        buf552 = buf546; del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_63, buf533, buf550, buf552, 104, 1568, grid=grid(104), stream=stream0)
        buf553 = buf545; del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_63, buf533, buf550, convolution_105, unsqueeze_1450, buf553, 1352, 121, grid=grid(1352), stream=stream0)
        buf554 = empty((104, ), device='cuda', dtype=torch.float32)
        buf556 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf553, squeeze_316, buf554, buf556, 104, 13, grid=grid(104), stream=stream0)
        buf555 = buf547; del buf547  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_63, buf533, buf550, convolution_105, unsqueeze_1450, buf554, squeeze_316, buf552, primals_317, buf555, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_105
        del le_63
        del primals_317
        del squeeze_316
        del unsqueeze_1450
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf557 = aten.convolution_backward(buf555, getitem_616, primals_316, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf555
        del getitem_616
        del primals_316
        buf558 = buf557[0]
        buf559 = buf557[1]
        del buf557
        buf560 = buf533; del buf533  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf560, le_64, buf558, buf550, buf542, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf542
        del buf550
        del le_64
        buf561 = buf520; del buf520  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf560, buf561, 416, 1568, grid=grid(416), stream=stream0)
        buf562 = buf519; del buf519  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf560, convolution_104, unsqueeze_1462, buf562, 5408, 121, grid=grid(5408), stream=stream0)
        buf563 = empty((416, ), device='cuda', dtype=torch.float32)
        buf564 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf562, squeeze_313, buf563, buf564, 416, 13, grid=grid(416), stream=stream0)
        buf565 = buf522; del buf522  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf560, convolution_104, unsqueeze_1462, buf563, squeeze_313, buf561, primals_314, buf565, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf560
        del convolution_104
        del primals_314
        del squeeze_313
        del unsqueeze_1462
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf566 = aten.convolution_backward(buf565, relu_100, primals_313, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_313
        buf567 = buf566[0]
        buf568 = buf566[1]
        del buf566
        buf569 = buf529; del buf529  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_100, buf526, buf567, buf569, 1024, 1568, grid=grid(1024), stream=stream0)
        buf570 = buf528; del buf528  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_100, buf526, buf567, convolution_103, unsqueeze_1474, buf570, 13312, 121, grid=grid(13312), stream=stream0)
        buf571 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf573 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf570, squeeze_310, buf571, buf573, 1024, 13, grid=grid(1024), stream=stream0)
        buf572 = buf531; del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_38.run(relu_100, buf526, buf567, convolution_103, unsqueeze_1474, buf571, squeeze_310, buf569, primals_311, buf572, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_103
        del primals_311
        del squeeze_310
        del unsqueeze_1474
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf574 = aten.convolution_backward(buf572, cat_19, primals_310, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf572
        del cat_19
        del primals_310
        buf575 = buf574[0]
        buf576 = buf574[1]
        del buf574
        buf577 = buf553; del buf553  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_66, buf575, buf577, 1352, 121, grid=grid(1352), stream=stream0)
        buf578 = buf554; del buf554  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf577, buf578, 104, 13, grid=grid(104), stream=stream0)
        buf579 = reinterpret_tensor(buf577, (104, 13), (1, 104), 0); del buf577  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_66, buf575, convolution_102, unsqueeze_1486, buf579, 1352, 121, grid=grid(1352), stream=stream0)
        buf580 = empty((104, ), device='cuda', dtype=torch.float32)
        buf581 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf579, squeeze_307, buf580, buf581, 104, 13, grid=grid(104), stream=stream0)
        buf582 = reinterpret_tensor(buf558, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf558  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_66, buf575, convolution_102, unsqueeze_1486, buf580, squeeze_307, buf578, primals_308, buf582, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_102
        del le_66
        del primals_308
        del squeeze_307
        del unsqueeze_1486
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf583 = aten.convolution_backward(buf582, add_562, primals_307, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_562
        del primals_307
        buf584 = buf583[0]
        buf585 = buf583[1]
        del buf583
        buf586 = buf580; del buf580  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_67, buf575, buf584, buf586, 104, 1568, grid=grid(104), stream=stream0)
        buf587 = reinterpret_tensor(buf579, (104, 13), (13, 1), 0); del buf579  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_67, buf575, buf584, convolution_101, unsqueeze_1498, buf587, 1352, 121, grid=grid(1352), stream=stream0)
        buf588 = empty((104, ), device='cuda', dtype=torch.float32)
        buf590 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf587, squeeze_304, buf588, buf590, 104, 13, grid=grid(104), stream=stream0)
        buf589 = buf582; del buf582  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_67, buf575, buf584, convolution_101, unsqueeze_1498, buf588, squeeze_304, buf586, primals_305, buf589, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_101
        del le_67
        del primals_305
        del squeeze_304
        del unsqueeze_1498
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf591 = aten.convolution_backward(buf589, add_556, primals_304, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_556
        del primals_304
        buf592 = buf591[0]
        buf593 = buf591[1]
        del buf591
        buf594 = buf588; del buf588  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_68, buf575, buf592, buf594, 104, 1568, grid=grid(104), stream=stream0)
        buf595 = buf587; del buf587  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_68, buf575, buf592, convolution_100, unsqueeze_1510, buf595, 1352, 121, grid=grid(1352), stream=stream0)
        buf596 = empty((104, ), device='cuda', dtype=torch.float32)
        buf598 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf595, squeeze_301, buf596, buf598, 104, 13, grid=grid(104), stream=stream0)
        buf597 = buf589; del buf589  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_68, buf575, buf592, convolution_100, unsqueeze_1510, buf596, squeeze_301, buf594, primals_302, buf597, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_100
        del le_68
        del primals_302
        del squeeze_301
        del unsqueeze_1510
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf599 = aten.convolution_backward(buf597, getitem_586, primals_301, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf597
        del getitem_586
        del primals_301
        buf600 = buf599[0]
        buf601 = buf599[1]
        del buf599
        buf602 = buf575; del buf575  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf602, le_69, buf600, buf592, buf584, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf584
        del buf592
        del le_69
        buf603 = buf563; del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf602, buf603, 416, 1568, grid=grid(416), stream=stream0)
        buf604 = buf562; del buf562  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf602, convolution_99, unsqueeze_1522, buf604, 5408, 121, grid=grid(5408), stream=stream0)
        buf605 = empty((416, ), device='cuda', dtype=torch.float32)
        buf606 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf604, squeeze_298, buf605, buf606, 416, 13, grid=grid(416), stream=stream0)
        buf607 = buf565; del buf565  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf602, convolution_99, unsqueeze_1522, buf605, squeeze_298, buf603, primals_299, buf607, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf602
        del convolution_99
        del primals_299
        del squeeze_298
        del unsqueeze_1522
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf608 = aten.convolution_backward(buf607, relu_95, primals_298, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_298
        buf609 = buf608[0]
        buf610 = buf608[1]
        del buf608
        buf611 = buf526; del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_60.run(buf611, relu_95, relu_100, buf567, buf609, 8192, 196, grid=grid(8192, 196), stream=stream0)
        del buf567
        del relu_100
        del relu_95
        buf612 = buf571; del buf571  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_57.run(buf611, buf612, 1024, 1568, grid=grid(1024), stream=stream0)
        buf613 = buf570; del buf570  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_58.run(buf611, convolution_98, unsqueeze_1534, buf613, 13312, 121, grid=grid(13312), stream=stream0)
        buf614 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf615 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf613, squeeze_295, buf614, buf615, 1024, 13, grid=grid(1024), stream=stream0)
        buf616 = reinterpret_tensor(buf609, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf609  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_59.run(buf611, convolution_98, unsqueeze_1534, buf614, squeeze_295, buf612, primals_296, buf616, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_98
        del primals_296
        del squeeze_295
        del unsqueeze_1534
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf617 = aten.convolution_backward(buf616, cat_18, primals_295, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_18
        del primals_295
        buf618 = buf617[0]
        buf619 = buf617[1]
        del buf617
        buf620 = buf595; del buf595  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_71, buf618, buf620, 1352, 121, grid=grid(1352), stream=stream0)
        buf621 = buf596; del buf596  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf620, buf621, 104, 13, grid=grid(104), stream=stream0)
        buf622 = reinterpret_tensor(buf620, (104, 13), (1, 104), 0); del buf620  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_71, buf618, convolution_97, unsqueeze_1546, buf622, 1352, 121, grid=grid(1352), stream=stream0)
        buf623 = empty((104, ), device='cuda', dtype=torch.float32)
        buf624 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf622, squeeze_292, buf623, buf624, 104, 13, grid=grid(104), stream=stream0)
        buf625 = reinterpret_tensor(buf600, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf600  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_71, buf618, convolution_97, unsqueeze_1546, buf623, squeeze_292, buf621, primals_293, buf625, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_97
        del le_71
        del primals_293
        del squeeze_292
        del unsqueeze_1546
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf626 = aten.convolution_backward(buf625, add_534, primals_292, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_534
        del primals_292
        buf627 = buf626[0]
        buf628 = buf626[1]
        del buf626
        buf629 = buf623; del buf623  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_72, buf618, buf627, buf629, 104, 1568, grid=grid(104), stream=stream0)
        buf630 = reinterpret_tensor(buf622, (104, 13), (13, 1), 0); del buf622  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_72, buf618, buf627, convolution_96, unsqueeze_1558, buf630, 1352, 121, grid=grid(1352), stream=stream0)
        buf631 = empty((104, ), device='cuda', dtype=torch.float32)
        buf633 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf630, squeeze_289, buf631, buf633, 104, 13, grid=grid(104), stream=stream0)
        buf632 = buf625; del buf625  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_72, buf618, buf627, convolution_96, unsqueeze_1558, buf631, squeeze_289, buf629, primals_290, buf632, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_96
        del le_72
        del primals_290
        del squeeze_289
        del unsqueeze_1558
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf634 = aten.convolution_backward(buf632, add_528, primals_289, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_528
        del primals_289
        buf635 = buf634[0]
        buf636 = buf634[1]
        del buf634
        buf637 = buf631; del buf631  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_73, buf618, buf635, buf637, 104, 1568, grid=grid(104), stream=stream0)
        buf638 = buf630; del buf630  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_73, buf618, buf635, convolution_95, unsqueeze_1570, buf638, 1352, 121, grid=grid(1352), stream=stream0)
        buf639 = empty((104, ), device='cuda', dtype=torch.float32)
        buf641 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf638, squeeze_286, buf639, buf641, 104, 13, grid=grid(104), stream=stream0)
        buf640 = buf632; del buf632  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_73, buf618, buf635, convolution_95, unsqueeze_1570, buf639, squeeze_286, buf637, primals_287, buf640, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_95
        del le_73
        del primals_287
        del squeeze_286
        del unsqueeze_1570
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf642 = aten.convolution_backward(buf640, getitem_556, primals_286, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf640
        del getitem_556
        del primals_286
        buf643 = buf642[0]
        buf644 = buf642[1]
        del buf642
        buf645 = buf618; del buf618  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf645, le_74, buf643, buf635, buf627, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf627
        del buf635
        del le_74
        buf646 = buf605; del buf605  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf645, buf646, 416, 1568, grid=grid(416), stream=stream0)
        buf647 = buf604; del buf604  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf645, convolution_94, unsqueeze_1582, buf647, 5408, 121, grid=grid(5408), stream=stream0)
        buf648 = empty((416, ), device='cuda', dtype=torch.float32)
        buf649 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf647, squeeze_283, buf648, buf649, 416, 13, grid=grid(416), stream=stream0)
        buf650 = buf607; del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf645, convolution_94, unsqueeze_1582, buf648, squeeze_283, buf646, primals_284, buf650, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf645
        del convolution_94
        del primals_284
        del squeeze_283
        del unsqueeze_1582
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf651 = aten.convolution_backward(buf650, relu_90, primals_283, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_283
        buf652 = buf651[0]
        buf653 = buf651[1]
        del buf651
        buf654 = buf614; del buf614  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_90, buf611, buf652, buf654, 1024, 1568, grid=grid(1024), stream=stream0)
        buf655 = buf613; del buf613  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_90, buf611, buf652, convolution_93, unsqueeze_1594, buf655, 13312, 121, grid=grid(13312), stream=stream0)
        buf656 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf658 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf655, squeeze_280, buf656, buf658, 1024, 13, grid=grid(1024), stream=stream0)
        buf657 = buf616; del buf616  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_38.run(relu_90, buf611, buf652, convolution_93, unsqueeze_1594, buf656, squeeze_280, buf654, primals_281, buf657, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_93
        del primals_281
        del squeeze_280
        del unsqueeze_1594
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf659 = aten.convolution_backward(buf657, cat_17, primals_280, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf657
        del cat_17
        del primals_280
        buf660 = buf659[0]
        buf661 = buf659[1]
        del buf659
        buf662 = buf638; del buf638  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_76, buf660, buf662, 1352, 121, grid=grid(1352), stream=stream0)
        buf663 = buf639; del buf639  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf662, buf663, 104, 13, grid=grid(104), stream=stream0)
        buf664 = reinterpret_tensor(buf662, (104, 13), (1, 104), 0); del buf662  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_76, buf660, convolution_92, unsqueeze_1606, buf664, 1352, 121, grid=grid(1352), stream=stream0)
        buf665 = empty((104, ), device='cuda', dtype=torch.float32)
        buf666 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf664, squeeze_277, buf665, buf666, 104, 13, grid=grid(104), stream=stream0)
        buf667 = reinterpret_tensor(buf643, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf643  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_76, buf660, convolution_92, unsqueeze_1606, buf665, squeeze_277, buf663, primals_278, buf667, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_92
        del le_76
        del primals_278
        del squeeze_277
        del unsqueeze_1606
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf668 = aten.convolution_backward(buf667, add_506, primals_277, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_506
        del primals_277
        buf669 = buf668[0]
        buf670 = buf668[1]
        del buf668
        buf671 = buf665; del buf665  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_77, buf660, buf669, buf671, 104, 1568, grid=grid(104), stream=stream0)
        buf672 = reinterpret_tensor(buf664, (104, 13), (13, 1), 0); del buf664  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_77, buf660, buf669, convolution_91, unsqueeze_1618, buf672, 1352, 121, grid=grid(1352), stream=stream0)
        buf673 = empty((104, ), device='cuda', dtype=torch.float32)
        buf675 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf672, squeeze_274, buf673, buf675, 104, 13, grid=grid(104), stream=stream0)
        buf674 = buf667; del buf667  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_77, buf660, buf669, convolution_91, unsqueeze_1618, buf673, squeeze_274, buf671, primals_275, buf674, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_91
        del le_77
        del primals_275
        del squeeze_274
        del unsqueeze_1618
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf676 = aten.convolution_backward(buf674, add_500, primals_274, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_500
        del primals_274
        buf677 = buf676[0]
        buf678 = buf676[1]
        del buf676
        buf679 = buf673; del buf673  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_78, buf660, buf677, buf679, 104, 1568, grid=grid(104), stream=stream0)
        buf680 = buf672; del buf672  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_78, buf660, buf677, convolution_90, unsqueeze_1630, buf680, 1352, 121, grid=grid(1352), stream=stream0)
        buf681 = empty((104, ), device='cuda', dtype=torch.float32)
        buf683 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf680, squeeze_271, buf681, buf683, 104, 13, grid=grid(104), stream=stream0)
        buf682 = buf674; del buf674  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_78, buf660, buf677, convolution_90, unsqueeze_1630, buf681, squeeze_271, buf679, primals_272, buf682, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_90
        del le_78
        del primals_272
        del squeeze_271
        del unsqueeze_1630
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf684 = aten.convolution_backward(buf682, getitem_526, primals_271, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf682
        del getitem_526
        del primals_271
        buf685 = buf684[0]
        buf686 = buf684[1]
        del buf684
        buf687 = buf660; del buf660  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf687, le_79, buf685, buf677, buf669, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf669
        del buf677
        del le_79
        buf688 = buf648; del buf648  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf687, buf688, 416, 1568, grid=grid(416), stream=stream0)
        buf689 = buf647; del buf647  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf687, convolution_89, unsqueeze_1642, buf689, 5408, 121, grid=grid(5408), stream=stream0)
        buf690 = empty((416, ), device='cuda', dtype=torch.float32)
        buf691 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf689, squeeze_268, buf690, buf691, 416, 13, grid=grid(416), stream=stream0)
        buf692 = buf650; del buf650  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf687, convolution_89, unsqueeze_1642, buf690, squeeze_268, buf688, primals_269, buf692, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf687
        del convolution_89
        del primals_269
        del squeeze_268
        del unsqueeze_1642
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf693 = aten.convolution_backward(buf692, relu_85, primals_268, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_268
        buf694 = buf693[0]
        buf695 = buf693[1]
        del buf693
        buf696 = buf611; del buf611  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_60.run(buf696, relu_85, relu_90, buf652, buf694, 8192, 196, grid=grid(8192, 196), stream=stream0)
        del buf652
        del relu_85
        del relu_90
        buf697 = buf656; del buf656  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_57.run(buf696, buf697, 1024, 1568, grid=grid(1024), stream=stream0)
        buf698 = buf655; del buf655  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_58.run(buf696, convolution_88, unsqueeze_1654, buf698, 13312, 121, grid=grid(13312), stream=stream0)
        buf699 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf700 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf698, squeeze_265, buf699, buf700, 1024, 13, grid=grid(1024), stream=stream0)
        buf701 = reinterpret_tensor(buf694, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf694  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_59.run(buf696, convolution_88, unsqueeze_1654, buf699, squeeze_265, buf697, primals_266, buf701, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_88
        del primals_266
        del squeeze_265
        del unsqueeze_1654
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf702 = aten.convolution_backward(buf701, cat_16, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_16
        del primals_265
        buf703 = buf702[0]
        buf704 = buf702[1]
        del buf702
        buf705 = buf680; del buf680  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_81, buf703, buf705, 1352, 121, grid=grid(1352), stream=stream0)
        buf706 = buf681; del buf681  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf705, buf706, 104, 13, grid=grid(104), stream=stream0)
        buf707 = reinterpret_tensor(buf705, (104, 13), (1, 104), 0); del buf705  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_81, buf703, convolution_87, unsqueeze_1666, buf707, 1352, 121, grid=grid(1352), stream=stream0)
        buf708 = empty((104, ), device='cuda', dtype=torch.float32)
        buf709 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf707, squeeze_262, buf708, buf709, 104, 13, grid=grid(104), stream=stream0)
        buf710 = reinterpret_tensor(buf685, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf685  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_81, buf703, convolution_87, unsqueeze_1666, buf708, squeeze_262, buf706, primals_263, buf710, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_87
        del le_81
        del primals_263
        del squeeze_262
        del unsqueeze_1666
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf711 = aten.convolution_backward(buf710, add_478, primals_262, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_478
        del primals_262
        buf712 = buf711[0]
        buf713 = buf711[1]
        del buf711
        buf714 = buf708; del buf708  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_82, buf703, buf712, buf714, 104, 1568, grid=grid(104), stream=stream0)
        buf715 = reinterpret_tensor(buf707, (104, 13), (13, 1), 0); del buf707  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_82, buf703, buf712, convolution_86, unsqueeze_1678, buf715, 1352, 121, grid=grid(1352), stream=stream0)
        buf716 = empty((104, ), device='cuda', dtype=torch.float32)
        buf718 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf715, squeeze_259, buf716, buf718, 104, 13, grid=grid(104), stream=stream0)
        buf717 = buf710; del buf710  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_82, buf703, buf712, convolution_86, unsqueeze_1678, buf716, squeeze_259, buf714, primals_260, buf717, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_86
        del le_82
        del primals_260
        del squeeze_259
        del unsqueeze_1678
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf719 = aten.convolution_backward(buf717, add_472, primals_259, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_472
        del primals_259
        buf720 = buf719[0]
        buf721 = buf719[1]
        del buf719
        buf722 = buf716; del buf716  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_83, buf703, buf720, buf722, 104, 1568, grid=grid(104), stream=stream0)
        buf723 = buf715; del buf715  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_83, buf703, buf720, convolution_85, unsqueeze_1690, buf723, 1352, 121, grid=grid(1352), stream=stream0)
        buf724 = empty((104, ), device='cuda', dtype=torch.float32)
        buf726 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf723, squeeze_256, buf724, buf726, 104, 13, grid=grid(104), stream=stream0)
        buf725 = buf717; del buf717  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_83, buf703, buf720, convolution_85, unsqueeze_1690, buf724, squeeze_256, buf722, primals_257, buf725, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_85
        del le_83
        del primals_257
        del squeeze_256
        del unsqueeze_1690
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf727 = aten.convolution_backward(buf725, getitem_496, primals_256, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf725
        del getitem_496
        del primals_256
        buf728 = buf727[0]
        buf729 = buf727[1]
        del buf727
        buf730 = buf703; del buf703  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf730, le_84, buf728, buf720, buf712, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf712
        del buf720
        del le_84
        buf731 = buf690; del buf690  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf730, buf731, 416, 1568, grid=grid(416), stream=stream0)
        buf732 = buf689; del buf689  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf730, convolution_84, unsqueeze_1702, buf732, 5408, 121, grid=grid(5408), stream=stream0)
        buf733 = empty((416, ), device='cuda', dtype=torch.float32)
        buf734 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf732, squeeze_253, buf733, buf734, 416, 13, grid=grid(416), stream=stream0)
        buf735 = buf692; del buf692  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf730, convolution_84, unsqueeze_1702, buf733, squeeze_253, buf731, primals_254, buf735, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf730
        del convolution_84
        del primals_254
        del squeeze_253
        del unsqueeze_1702
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf736 = aten.convolution_backward(buf735, relu_80, primals_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_253
        buf737 = buf736[0]
        buf738 = buf736[1]
        del buf736
        buf739 = buf699; del buf699  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_80, buf696, buf737, buf739, 1024, 1568, grid=grid(1024), stream=stream0)
        buf740 = buf698; del buf698  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_80, buf696, buf737, convolution_83, unsqueeze_1714, buf740, 13312, 121, grid=grid(13312), stream=stream0)
        buf741 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf743 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf740, squeeze_250, buf741, buf743, 1024, 13, grid=grid(1024), stream=stream0)
        buf742 = buf701; del buf701  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_38.run(relu_80, buf696, buf737, convolution_83, unsqueeze_1714, buf741, squeeze_250, buf739, primals_251, buf742, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_83
        del primals_251
        del squeeze_250
        del unsqueeze_1714
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf744 = aten.convolution_backward(buf742, cat_15, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf742
        del cat_15
        del primals_250
        buf745 = buf744[0]
        buf746 = buf744[1]
        del buf744
        buf747 = buf723; del buf723  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_86, buf745, buf747, 1352, 121, grid=grid(1352), stream=stream0)
        buf748 = buf724; del buf724  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf747, buf748, 104, 13, grid=grid(104), stream=stream0)
        buf749 = reinterpret_tensor(buf747, (104, 13), (1, 104), 0); del buf747  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_86, buf745, convolution_82, unsqueeze_1726, buf749, 1352, 121, grid=grid(1352), stream=stream0)
        buf750 = empty((104, ), device='cuda', dtype=torch.float32)
        buf751 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf749, squeeze_247, buf750, buf751, 104, 13, grid=grid(104), stream=stream0)
        buf752 = reinterpret_tensor(buf728, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf728  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_86, buf745, convolution_82, unsqueeze_1726, buf750, squeeze_247, buf748, primals_248, buf752, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_82
        del le_86
        del primals_248
        del squeeze_247
        del unsqueeze_1726
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf753 = aten.convolution_backward(buf752, add_450, primals_247, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_450
        del primals_247
        buf754 = buf753[0]
        buf755 = buf753[1]
        del buf753
        buf756 = buf750; del buf750  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_87, buf745, buf754, buf756, 104, 1568, grid=grid(104), stream=stream0)
        buf757 = reinterpret_tensor(buf749, (104, 13), (13, 1), 0); del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_87, buf745, buf754, convolution_81, unsqueeze_1738, buf757, 1352, 121, grid=grid(1352), stream=stream0)
        buf758 = empty((104, ), device='cuda', dtype=torch.float32)
        buf760 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf757, squeeze_244, buf758, buf760, 104, 13, grid=grid(104), stream=stream0)
        buf759 = buf752; del buf752  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_87, buf745, buf754, convolution_81, unsqueeze_1738, buf758, squeeze_244, buf756, primals_245, buf759, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_81
        del le_87
        del primals_245
        del squeeze_244
        del unsqueeze_1738
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf761 = aten.convolution_backward(buf759, add_444, primals_244, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_444
        del primals_244
        buf762 = buf761[0]
        buf763 = buf761[1]
        del buf761
        buf764 = buf758; del buf758  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_88, buf745, buf762, buf764, 104, 1568, grid=grid(104), stream=stream0)
        buf765 = buf757; del buf757  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_88, buf745, buf762, convolution_80, unsqueeze_1750, buf765, 1352, 121, grid=grid(1352), stream=stream0)
        buf766 = empty((104, ), device='cuda', dtype=torch.float32)
        buf768 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf765, squeeze_241, buf766, buf768, 104, 13, grid=grid(104), stream=stream0)
        buf767 = buf759; del buf759  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_88, buf745, buf762, convolution_80, unsqueeze_1750, buf766, squeeze_241, buf764, primals_242, buf767, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_80
        del le_88
        del primals_242
        del squeeze_241
        del unsqueeze_1750
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf769 = aten.convolution_backward(buf767, getitem_466, primals_241, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf767
        del getitem_466
        del primals_241
        buf770 = buf769[0]
        buf771 = buf769[1]
        del buf769
        buf772 = buf745; del buf745  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf772, le_89, buf770, buf762, buf754, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf754
        del buf762
        del le_89
        buf773 = buf733; del buf733  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf772, buf773, 416, 1568, grid=grid(416), stream=stream0)
        buf774 = buf732; del buf732  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf772, convolution_79, unsqueeze_1762, buf774, 5408, 121, grid=grid(5408), stream=stream0)
        buf775 = empty((416, ), device='cuda', dtype=torch.float32)
        buf776 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf774, squeeze_238, buf775, buf776, 416, 13, grid=grid(416), stream=stream0)
        buf777 = buf735; del buf735  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf772, convolution_79, unsqueeze_1762, buf775, squeeze_238, buf773, primals_239, buf777, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf772
        del convolution_79
        del primals_239
        del squeeze_238
        del unsqueeze_1762
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf778 = aten.convolution_backward(buf777, relu_75, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_238
        buf779 = buf778[0]
        buf780 = buf778[1]
        del buf778
        buf781 = buf696; del buf696  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_60.run(buf781, relu_75, relu_80, buf737, buf779, 8192, 196, grid=grid(8192, 196), stream=stream0)
        del buf737
        del relu_75
        del relu_80
        buf782 = buf741; del buf741  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_57.run(buf781, buf782, 1024, 1568, grid=grid(1024), stream=stream0)
        buf783 = buf740; del buf740  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_58.run(buf781, convolution_78, unsqueeze_1774, buf783, 13312, 121, grid=grid(13312), stream=stream0)
        buf784 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf785 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf783, squeeze_235, buf784, buf785, 1024, 13, grid=grid(1024), stream=stream0)
        buf786 = reinterpret_tensor(buf779, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf779  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_59.run(buf781, convolution_78, unsqueeze_1774, buf784, squeeze_235, buf782, primals_236, buf786, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_78
        del primals_236
        del squeeze_235
        del unsqueeze_1774
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf787 = aten.convolution_backward(buf786, cat_14, primals_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_14
        del primals_235
        buf788 = buf787[0]
        buf789 = buf787[1]
        del buf787
        buf790 = buf765; del buf765  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_91, buf788, buf790, 1352, 121, grid=grid(1352), stream=stream0)
        buf791 = buf766; del buf766  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf790, buf791, 104, 13, grid=grid(104), stream=stream0)
        buf792 = reinterpret_tensor(buf790, (104, 13), (1, 104), 0); del buf790  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_91, buf788, convolution_77, unsqueeze_1786, buf792, 1352, 121, grid=grid(1352), stream=stream0)
        buf793 = empty((104, ), device='cuda', dtype=torch.float32)
        buf794 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf792, squeeze_232, buf793, buf794, 104, 13, grid=grid(104), stream=stream0)
        buf795 = reinterpret_tensor(buf770, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf770  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_91, buf788, convolution_77, unsqueeze_1786, buf793, squeeze_232, buf791, primals_233, buf795, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_77
        del le_91
        del primals_233
        del squeeze_232
        del unsqueeze_1786
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf796 = aten.convolution_backward(buf795, add_422, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_422
        del primals_232
        buf797 = buf796[0]
        buf798 = buf796[1]
        del buf796
        buf799 = buf793; del buf793  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_92, buf788, buf797, buf799, 104, 1568, grid=grid(104), stream=stream0)
        buf800 = reinterpret_tensor(buf792, (104, 13), (13, 1), 0); del buf792  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_92, buf788, buf797, convolution_76, unsqueeze_1798, buf800, 1352, 121, grid=grid(1352), stream=stream0)
        buf801 = empty((104, ), device='cuda', dtype=torch.float32)
        buf803 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf800, squeeze_229, buf801, buf803, 104, 13, grid=grid(104), stream=stream0)
        buf802 = buf795; del buf795  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_92, buf788, buf797, convolution_76, unsqueeze_1798, buf801, squeeze_229, buf799, primals_230, buf802, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_76
        del le_92
        del primals_230
        del squeeze_229
        del unsqueeze_1798
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf804 = aten.convolution_backward(buf802, add_416, primals_229, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_416
        del primals_229
        buf805 = buf804[0]
        buf806 = buf804[1]
        del buf804
        buf807 = buf801; del buf801  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_93, buf788, buf805, buf807, 104, 1568, grid=grid(104), stream=stream0)
        buf808 = buf800; del buf800  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_93, buf788, buf805, convolution_75, unsqueeze_1810, buf808, 1352, 121, grid=grid(1352), stream=stream0)
        buf809 = empty((104, ), device='cuda', dtype=torch.float32)
        buf811 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf808, squeeze_226, buf809, buf811, 104, 13, grid=grid(104), stream=stream0)
        buf810 = buf802; del buf802  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_93, buf788, buf805, convolution_75, unsqueeze_1810, buf809, squeeze_226, buf807, primals_227, buf810, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_75
        del le_93
        del primals_227
        del squeeze_226
        del unsqueeze_1810
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf812 = aten.convolution_backward(buf810, getitem_436, primals_226, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf810
        del getitem_436
        del primals_226
        buf813 = buf812[0]
        buf814 = buf812[1]
        del buf812
        buf815 = buf788; del buf788  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf815, le_94, buf813, buf805, buf797, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf797
        del buf805
        del le_94
        buf816 = buf775; del buf775  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf815, buf816, 416, 1568, grid=grid(416), stream=stream0)
        buf817 = buf774; del buf774  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf815, convolution_74, unsqueeze_1822, buf817, 5408, 121, grid=grid(5408), stream=stream0)
        buf818 = empty((416, ), device='cuda', dtype=torch.float32)
        buf819 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf817, squeeze_223, buf818, buf819, 416, 13, grid=grid(416), stream=stream0)
        buf820 = buf777; del buf777  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf815, convolution_74, unsqueeze_1822, buf818, squeeze_223, buf816, primals_224, buf820, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf815
        del convolution_74
        del primals_224
        del squeeze_223
        del unsqueeze_1822
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf821 = aten.convolution_backward(buf820, relu_70, primals_223, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_223
        buf822 = buf821[0]
        buf823 = buf821[1]
        del buf821
        buf824 = buf784; del buf784  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_70, buf781, buf822, buf824, 1024, 1568, grid=grid(1024), stream=stream0)
        buf825 = buf783; del buf783  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_70, buf781, buf822, convolution_73, unsqueeze_1834, buf825, 13312, 121, grid=grid(13312), stream=stream0)
        buf826 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf828 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf825, squeeze_220, buf826, buf828, 1024, 13, grid=grid(1024), stream=stream0)
        buf827 = buf786; del buf786  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_38.run(relu_70, buf781, buf822, convolution_73, unsqueeze_1834, buf826, squeeze_220, buf824, primals_221, buf827, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_73
        del primals_221
        del squeeze_220
        del unsqueeze_1834
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf829 = aten.convolution_backward(buf827, cat_13, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf827
        del cat_13
        del primals_220
        buf830 = buf829[0]
        buf831 = buf829[1]
        del buf829
        buf832 = buf808; del buf808  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_96, buf830, buf832, 1352, 121, grid=grid(1352), stream=stream0)
        buf833 = buf809; del buf809  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf832, buf833, 104, 13, grid=grid(104), stream=stream0)
        buf834 = reinterpret_tensor(buf832, (104, 13), (1, 104), 0); del buf832  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_96, buf830, convolution_72, unsqueeze_1846, buf834, 1352, 121, grid=grid(1352), stream=stream0)
        buf835 = empty((104, ), device='cuda', dtype=torch.float32)
        buf836 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf834, squeeze_217, buf835, buf836, 104, 13, grid=grid(104), stream=stream0)
        buf837 = reinterpret_tensor(buf813, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf813  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_96, buf830, convolution_72, unsqueeze_1846, buf835, squeeze_217, buf833, primals_218, buf837, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_72
        del le_96
        del primals_218
        del squeeze_217
        del unsqueeze_1846
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf838 = aten.convolution_backward(buf837, add_394, primals_217, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_394
        del primals_217
        buf839 = buf838[0]
        buf840 = buf838[1]
        del buf838
        buf841 = buf835; del buf835  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_97, buf830, buf839, buf841, 104, 1568, grid=grid(104), stream=stream0)
        buf842 = reinterpret_tensor(buf834, (104, 13), (13, 1), 0); del buf834  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_97, buf830, buf839, convolution_71, unsqueeze_1858, buf842, 1352, 121, grid=grid(1352), stream=stream0)
        buf843 = empty((104, ), device='cuda', dtype=torch.float32)
        buf845 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf842, squeeze_214, buf843, buf845, 104, 13, grid=grid(104), stream=stream0)
        buf844 = buf837; del buf837  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_97, buf830, buf839, convolution_71, unsqueeze_1858, buf843, squeeze_214, buf841, primals_215, buf844, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_71
        del le_97
        del primals_215
        del squeeze_214
        del unsqueeze_1858
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf846 = aten.convolution_backward(buf844, add_388, primals_214, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_388
        del primals_214
        buf847 = buf846[0]
        buf848 = buf846[1]
        del buf846
        buf849 = buf843; del buf843  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_98, buf830, buf847, buf849, 104, 1568, grid=grid(104), stream=stream0)
        buf850 = buf842; del buf842  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_98, buf830, buf847, convolution_70, unsqueeze_1870, buf850, 1352, 121, grid=grid(1352), stream=stream0)
        buf851 = empty((104, ), device='cuda', dtype=torch.float32)
        buf853 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf850, squeeze_211, buf851, buf853, 104, 13, grid=grid(104), stream=stream0)
        buf852 = buf844; del buf844  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_98, buf830, buf847, convolution_70, unsqueeze_1870, buf851, squeeze_211, buf849, primals_212, buf852, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_70
        del le_98
        del primals_212
        del squeeze_211
        del unsqueeze_1870
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf854 = aten.convolution_backward(buf852, getitem_406, primals_211, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf852
        del getitem_406
        del primals_211
        buf855 = buf854[0]
        buf856 = buf854[1]
        del buf854
        buf857 = buf830; del buf830  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf857, le_99, buf855, buf847, buf839, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf839
        del buf847
        del le_99
        buf858 = buf818; del buf818  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf857, buf858, 416, 1568, grid=grid(416), stream=stream0)
        buf859 = buf817; del buf817  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf857, convolution_69, unsqueeze_1882, buf859, 5408, 121, grid=grid(5408), stream=stream0)
        buf860 = empty((416, ), device='cuda', dtype=torch.float32)
        buf861 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf859, squeeze_208, buf860, buf861, 416, 13, grid=grid(416), stream=stream0)
        buf862 = buf820; del buf820  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf857, convolution_69, unsqueeze_1882, buf860, squeeze_208, buf858, primals_209, buf862, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf857
        del convolution_69
        del primals_209
        del squeeze_208
        del unsqueeze_1882
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf863 = aten.convolution_backward(buf862, relu_65, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_208
        buf864 = buf863[0]
        buf865 = buf863[1]
        del buf863
        buf866 = buf781; del buf781  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_60.run(buf866, relu_65, relu_70, buf822, buf864, 8192, 196, grid=grid(8192, 196), stream=stream0)
        del buf822
        del relu_65
        del relu_70
        buf867 = buf826; del buf826  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_57.run(buf866, buf867, 1024, 1568, grid=grid(1024), stream=stream0)
        buf868 = buf825; del buf825  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_58.run(buf866, convolution_68, unsqueeze_1894, buf868, 13312, 121, grid=grid(13312), stream=stream0)
        buf869 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf870 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf868, squeeze_205, buf869, buf870, 1024, 13, grid=grid(1024), stream=stream0)
        buf871 = reinterpret_tensor(buf864, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf864  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_59.run(buf866, convolution_68, unsqueeze_1894, buf869, squeeze_205, buf867, primals_206, buf871, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_68
        del primals_206
        del squeeze_205
        del unsqueeze_1894
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf872 = aten.convolution_backward(buf871, cat_12, primals_205, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_12
        del primals_205
        buf873 = buf872[0]
        buf874 = buf872[1]
        del buf872
        buf875 = buf850; del buf850  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_101, buf873, buf875, 1352, 121, grid=grid(1352), stream=stream0)
        buf876 = buf851; del buf851  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf875, buf876, 104, 13, grid=grid(104), stream=stream0)
        buf877 = reinterpret_tensor(buf875, (104, 13), (1, 104), 0); del buf875  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_101, buf873, convolution_67, unsqueeze_1906, buf877, 1352, 121, grid=grid(1352), stream=stream0)
        buf878 = empty((104, ), device='cuda', dtype=torch.float32)
        buf879 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf877, squeeze_202, buf878, buf879, 104, 13, grid=grid(104), stream=stream0)
        buf880 = reinterpret_tensor(buf855, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf855  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_101, buf873, convolution_67, unsqueeze_1906, buf878, squeeze_202, buf876, primals_203, buf880, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_67
        del le_101
        del primals_203
        del squeeze_202
        del unsqueeze_1906
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf881 = aten.convolution_backward(buf880, add_366, primals_202, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_366
        del primals_202
        buf882 = buf881[0]
        buf883 = buf881[1]
        del buf881
        buf884 = buf878; del buf878  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_102, buf873, buf882, buf884, 104, 1568, grid=grid(104), stream=stream0)
        buf885 = reinterpret_tensor(buf877, (104, 13), (13, 1), 0); del buf877  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_102, buf873, buf882, convolution_66, unsqueeze_1918, buf885, 1352, 121, grid=grid(1352), stream=stream0)
        buf886 = empty((104, ), device='cuda', dtype=torch.float32)
        buf888 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf885, squeeze_199, buf886, buf888, 104, 13, grid=grid(104), stream=stream0)
        buf887 = buf880; del buf880  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_102, buf873, buf882, convolution_66, unsqueeze_1918, buf886, squeeze_199, buf884, primals_200, buf887, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_66
        del le_102
        del primals_200
        del squeeze_199
        del unsqueeze_1918
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf889 = aten.convolution_backward(buf887, add_360, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_360
        del primals_199
        buf890 = buf889[0]
        buf891 = buf889[1]
        del buf889
        buf892 = buf886; del buf886  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_103, buf873, buf890, buf892, 104, 1568, grid=grid(104), stream=stream0)
        buf893 = buf885; del buf885  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_103, buf873, buf890, convolution_65, unsqueeze_1930, buf893, 1352, 121, grid=grid(1352), stream=stream0)
        buf894 = empty((104, ), device='cuda', dtype=torch.float32)
        buf896 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf893, squeeze_196, buf894, buf896, 104, 13, grid=grid(104), stream=stream0)
        buf895 = buf887; del buf887  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_103, buf873, buf890, convolution_65, unsqueeze_1930, buf894, squeeze_196, buf892, primals_197, buf895, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_65
        del le_103
        del primals_197
        del squeeze_196
        del unsqueeze_1930
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf897 = aten.convolution_backward(buf895, getitem_376, primals_196, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf895
        del getitem_376
        del primals_196
        buf898 = buf897[0]
        buf899 = buf897[1]
        del buf897
        buf900 = buf873; del buf873  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf900, le_104, buf898, buf890, buf882, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf882
        del buf890
        del le_104
        buf901 = buf860; del buf860  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf900, buf901, 416, 1568, grid=grid(416), stream=stream0)
        buf902 = buf859; del buf859  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf900, convolution_64, unsqueeze_1942, buf902, 5408, 121, grid=grid(5408), stream=stream0)
        buf903 = empty((416, ), device='cuda', dtype=torch.float32)
        buf904 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf902, squeeze_193, buf903, buf904, 416, 13, grid=grid(416), stream=stream0)
        buf905 = buf862; del buf862  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf900, convolution_64, unsqueeze_1942, buf903, squeeze_193, buf901, primals_194, buf905, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf900
        del convolution_64
        del primals_194
        del squeeze_193
        del unsqueeze_1942
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf906 = aten.convolution_backward(buf905, relu_60, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_193
        buf907 = buf906[0]
        buf908 = buf906[1]
        del buf906
        buf909 = buf869; del buf869  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_60, buf866, buf907, buf909, 1024, 1568, grid=grid(1024), stream=stream0)
        buf910 = buf868; del buf868  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_60, buf866, buf907, convolution_63, unsqueeze_1954, buf910, 13312, 121, grid=grid(13312), stream=stream0)
        buf911 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf913 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf910, squeeze_190, buf911, buf913, 1024, 13, grid=grid(1024), stream=stream0)
        buf912 = buf871; del buf871  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_38.run(relu_60, buf866, buf907, convolution_63, unsqueeze_1954, buf911, squeeze_190, buf909, primals_191, buf912, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_63
        del primals_191
        del squeeze_190
        del unsqueeze_1954
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf914 = aten.convolution_backward(buf912, cat_11, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf912
        del cat_11
        del primals_190
        buf915 = buf914[0]
        buf916 = buf914[1]
        del buf914
        buf917 = buf893; del buf893  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_106, buf915, buf917, 1352, 121, grid=grid(1352), stream=stream0)
        buf918 = buf894; del buf894  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf917, buf918, 104, 13, grid=grid(104), stream=stream0)
        buf919 = reinterpret_tensor(buf917, (104, 13), (1, 104), 0); del buf917  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_106, buf915, convolution_62, unsqueeze_1966, buf919, 1352, 121, grid=grid(1352), stream=stream0)
        buf920 = empty((104, ), device='cuda', dtype=torch.float32)
        buf921 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf919, squeeze_187, buf920, buf921, 104, 13, grid=grid(104), stream=stream0)
        buf922 = reinterpret_tensor(buf898, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf898  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_106, buf915, convolution_62, unsqueeze_1966, buf920, squeeze_187, buf918, primals_188, buf922, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_62
        del le_106
        del primals_188
        del squeeze_187
        del unsqueeze_1966
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf923 = aten.convolution_backward(buf922, add_338, primals_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_338
        del primals_187
        buf924 = buf923[0]
        buf925 = buf923[1]
        del buf923
        buf926 = buf920; del buf920  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_107, buf915, buf924, buf926, 104, 1568, grid=grid(104), stream=stream0)
        buf927 = reinterpret_tensor(buf919, (104, 13), (13, 1), 0); del buf919  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_107, buf915, buf924, convolution_61, unsqueeze_1978, buf927, 1352, 121, grid=grid(1352), stream=stream0)
        buf928 = empty((104, ), device='cuda', dtype=torch.float32)
        buf930 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf927, squeeze_184, buf928, buf930, 104, 13, grid=grid(104), stream=stream0)
        buf929 = buf922; del buf922  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_107, buf915, buf924, convolution_61, unsqueeze_1978, buf928, squeeze_184, buf926, primals_185, buf929, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_61
        del le_107
        del primals_185
        del squeeze_184
        del unsqueeze_1978
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf931 = aten.convolution_backward(buf929, add_332, primals_184, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_332
        del primals_184
        buf932 = buf931[0]
        buf933 = buf931[1]
        del buf931
        buf934 = buf928; del buf928  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_108, buf915, buf932, buf934, 104, 1568, grid=grid(104), stream=stream0)
        buf935 = buf927; del buf927  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_108, buf915, buf932, convolution_60, unsqueeze_1990, buf935, 1352, 121, grid=grid(1352), stream=stream0)
        buf936 = empty((104, ), device='cuda', dtype=torch.float32)
        buf938 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf935, squeeze_181, buf936, buf938, 104, 13, grid=grid(104), stream=stream0)
        buf937 = buf929; del buf929  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_108, buf915, buf932, convolution_60, unsqueeze_1990, buf936, squeeze_181, buf934, primals_182, buf937, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_60
        del le_108
        del primals_182
        del squeeze_181
        del unsqueeze_1990
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf939 = aten.convolution_backward(buf937, getitem_346, primals_181, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf937
        del getitem_346
        del primals_181
        buf940 = buf939[0]
        buf941 = buf939[1]
        del buf939
        buf942 = buf915; del buf915  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf942, le_109, buf940, buf932, buf924, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf924
        del buf932
        del le_109
        buf943 = buf903; del buf903  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf942, buf943, 416, 1568, grid=grid(416), stream=stream0)
        buf944 = buf902; del buf902  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf942, convolution_59, unsqueeze_2002, buf944, 5408, 121, grid=grid(5408), stream=stream0)
        buf945 = empty((416, ), device='cuda', dtype=torch.float32)
        buf946 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf944, squeeze_178, buf945, buf946, 416, 13, grid=grid(416), stream=stream0)
        buf947 = buf905; del buf905  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf942, convolution_59, unsqueeze_2002, buf945, squeeze_178, buf943, primals_179, buf947, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf942
        del convolution_59
        del primals_179
        del squeeze_178
        del unsqueeze_2002
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf948 = aten.convolution_backward(buf947, relu_55, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_178
        buf949 = buf948[0]
        buf950 = buf948[1]
        del buf948
        buf951 = buf866; del buf866  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_60.run(buf951, relu_55, relu_60, buf907, buf949, 8192, 196, grid=grid(8192, 196), stream=stream0)
        del buf907
        del relu_55
        del relu_60
        buf952 = buf911; del buf911  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_57.run(buf951, buf952, 1024, 1568, grid=grid(1024), stream=stream0)
        buf953 = buf910; del buf910  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_58.run(buf951, convolution_58, unsqueeze_2014, buf953, 13312, 121, grid=grid(13312), stream=stream0)
        buf954 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf955 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf953, squeeze_175, buf954, buf955, 1024, 13, grid=grid(1024), stream=stream0)
        buf956 = reinterpret_tensor(buf949, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf949  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_59.run(buf951, convolution_58, unsqueeze_2014, buf954, squeeze_175, buf952, primals_176, buf956, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_58
        del primals_176
        del squeeze_175
        del unsqueeze_2014
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf957 = aten.convolution_backward(buf956, cat_10, primals_175, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_10
        del primals_175
        buf958 = buf957[0]
        buf959 = buf957[1]
        del buf957
        buf960 = buf935; del buf935  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_111, buf958, buf960, 1352, 121, grid=grid(1352), stream=stream0)
        buf961 = buf936; del buf936  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf960, buf961, 104, 13, grid=grid(104), stream=stream0)
        buf962 = reinterpret_tensor(buf960, (104, 13), (1, 104), 0); del buf960  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_111, buf958, convolution_57, unsqueeze_2026, buf962, 1352, 121, grid=grid(1352), stream=stream0)
        buf963 = empty((104, ), device='cuda', dtype=torch.float32)
        buf964 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf962, squeeze_172, buf963, buf964, 104, 13, grid=grid(104), stream=stream0)
        buf965 = reinterpret_tensor(buf940, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf940  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_111, buf958, convolution_57, unsqueeze_2026, buf963, squeeze_172, buf961, primals_173, buf965, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_57
        del le_111
        del primals_173
        del squeeze_172
        del unsqueeze_2026
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf966 = aten.convolution_backward(buf965, add_310, primals_172, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_310
        del primals_172
        buf967 = buf966[0]
        buf968 = buf966[1]
        del buf966
        buf969 = buf963; del buf963  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_112, buf958, buf967, buf969, 104, 1568, grid=grid(104), stream=stream0)
        buf970 = reinterpret_tensor(buf962, (104, 13), (13, 1), 0); del buf962  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_112, buf958, buf967, convolution_56, unsqueeze_2038, buf970, 1352, 121, grid=grid(1352), stream=stream0)
        buf971 = empty((104, ), device='cuda', dtype=torch.float32)
        buf973 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf970, squeeze_169, buf971, buf973, 104, 13, grid=grid(104), stream=stream0)
        buf972 = buf965; del buf965  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_112, buf958, buf967, convolution_56, unsqueeze_2038, buf971, squeeze_169, buf969, primals_170, buf972, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_56
        del le_112
        del primals_170
        del squeeze_169
        del unsqueeze_2038
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf974 = aten.convolution_backward(buf972, add_304, primals_169, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_304
        del primals_169
        buf975 = buf974[0]
        buf976 = buf974[1]
        del buf974
        buf977 = buf971; del buf971  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_113, buf958, buf975, buf977, 104, 1568, grid=grid(104), stream=stream0)
        buf978 = buf970; del buf970  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_113, buf958, buf975, convolution_55, unsqueeze_2050, buf978, 1352, 121, grid=grid(1352), stream=stream0)
        buf979 = empty((104, ), device='cuda', dtype=torch.float32)
        buf981 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf978, squeeze_166, buf979, buf981, 104, 13, grid=grid(104), stream=stream0)
        buf980 = buf972; del buf972  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_113, buf958, buf975, convolution_55, unsqueeze_2050, buf979, squeeze_166, buf977, primals_167, buf980, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_55
        del le_113
        del primals_167
        del squeeze_166
        del unsqueeze_2050
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf982 = aten.convolution_backward(buf980, getitem_316, primals_166, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf980
        del getitem_316
        del primals_166
        buf983 = buf982[0]
        buf984 = buf982[1]
        del buf982
        buf985 = buf958; del buf958  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf985, le_114, buf983, buf975, buf967, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf967
        del buf975
        del le_114
        buf986 = buf945; del buf945  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf985, buf986, 416, 1568, grid=grid(416), stream=stream0)
        buf987 = buf944; del buf944  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf985, convolution_54, unsqueeze_2062, buf987, 5408, 121, grid=grid(5408), stream=stream0)
        buf988 = empty((416, ), device='cuda', dtype=torch.float32)
        buf989 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf987, squeeze_163, buf988, buf989, 416, 13, grid=grid(416), stream=stream0)
        buf990 = buf947; del buf947  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf985, convolution_54, unsqueeze_2062, buf988, squeeze_163, buf986, primals_164, buf990, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf985
        del convolution_54
        del primals_164
        del squeeze_163
        del unsqueeze_2062
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf991 = aten.convolution_backward(buf990, relu_50, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_163
        buf992 = buf991[0]
        buf993 = buf991[1]
        del buf991
        buf994 = buf954; del buf954  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_50, buf951, buf992, buf994, 1024, 1568, grid=grid(1024), stream=stream0)
        buf995 = buf953; del buf953  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_36.run(relu_50, buf951, buf992, convolution_53, unsqueeze_2074, buf995, 13312, 121, grid=grid(13312), stream=stream0)
        buf996 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf998 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf995, squeeze_160, buf996, buf998, 1024, 13, grid=grid(1024), stream=stream0)
        buf997 = buf956; del buf956  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_38.run(relu_50, buf951, buf992, convolution_53, unsqueeze_2074, buf996, squeeze_160, buf994, primals_161, buf997, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_53
        del primals_161
        del squeeze_160
        del unsqueeze_2074
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf999 = aten.convolution_backward(buf997, cat_9, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf997
        del cat_9
        del primals_160
        buf1000 = buf999[0]
        buf1001 = buf999[1]
        del buf999
        buf1002 = buf978; del buf978  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_116, buf1000, buf1002, 1352, 121, grid=grid(1352), stream=stream0)
        buf1003 = buf979; del buf979  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf1002, buf1003, 104, 13, grid=grid(104), stream=stream0)
        buf1004 = reinterpret_tensor(buf1002, (104, 13), (1, 104), 0); del buf1002  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_116, buf1000, convolution_52, unsqueeze_2086, buf1004, 1352, 121, grid=grid(1352), stream=stream0)
        buf1005 = empty((104, ), device='cuda', dtype=torch.float32)
        buf1006 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf1004, squeeze_157, buf1005, buf1006, 104, 13, grid=grid(104), stream=stream0)
        buf1007 = reinterpret_tensor(buf983, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf983  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_116, buf1000, convolution_52, unsqueeze_2086, buf1005, squeeze_157, buf1003, primals_158, buf1007, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_52
        del le_116
        del primals_158
        del squeeze_157
        del unsqueeze_2086
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1008 = aten.convolution_backward(buf1007, add_282, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_282
        del primals_157
        buf1009 = buf1008[0]
        buf1010 = buf1008[1]
        del buf1008
        buf1011 = buf1005; del buf1005  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_117, buf1000, buf1009, buf1011, 104, 1568, grid=grid(104), stream=stream0)
        buf1012 = reinterpret_tensor(buf1004, (104, 13), (13, 1), 0); del buf1004  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_117, buf1000, buf1009, convolution_51, unsqueeze_2098, buf1012, 1352, 121, grid=grid(1352), stream=stream0)
        buf1013 = empty((104, ), device='cuda', dtype=torch.float32)
        buf1015 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf1012, squeeze_154, buf1013, buf1015, 104, 13, grid=grid(104), stream=stream0)
        buf1014 = buf1007; del buf1007  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_117, buf1000, buf1009, convolution_51, unsqueeze_2098, buf1013, squeeze_154, buf1011, primals_155, buf1014, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_51
        del le_117
        del primals_155
        del squeeze_154
        del unsqueeze_2098
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1016 = aten.convolution_backward(buf1014, add_276, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_276
        del primals_154
        buf1017 = buf1016[0]
        buf1018 = buf1016[1]
        del buf1016
        buf1019 = buf1013; del buf1013  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_118, buf1000, buf1017, buf1019, 104, 1568, grid=grid(104), stream=stream0)
        buf1020 = buf1012; del buf1012  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_118, buf1000, buf1017, convolution_50, unsqueeze_2110, buf1020, 1352, 121, grid=grid(1352), stream=stream0)
        buf1021 = empty((104, ), device='cuda', dtype=torch.float32)
        buf1023 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf1020, squeeze_151, buf1021, buf1023, 104, 13, grid=grid(104), stream=stream0)
        buf1022 = buf1014; del buf1014  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_118, buf1000, buf1017, convolution_50, unsqueeze_2110, buf1021, squeeze_151, buf1019, primals_152, buf1022, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_50
        del le_118
        del primals_152
        del squeeze_151
        del unsqueeze_2110
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1024 = aten.convolution_backward(buf1022, getitem_286, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1022
        del getitem_286
        del primals_151
        buf1025 = buf1024[0]
        buf1026 = buf1024[1]
        del buf1024
        buf1027 = buf1000; del buf1000  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf1027, le_119, buf1025, buf1017, buf1009, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf1009
        del buf1017
        del le_119
        buf1028 = buf988; del buf988  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf1027, buf1028, 416, 1568, grid=grid(416), stream=stream0)
        buf1029 = buf987; del buf987  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf1027, convolution_49, unsqueeze_2122, buf1029, 5408, 121, grid=grid(5408), stream=stream0)
        buf1030 = empty((416, ), device='cuda', dtype=torch.float32)
        buf1031 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf1029, squeeze_148, buf1030, buf1031, 416, 13, grid=grid(416), stream=stream0)
        buf1032 = buf990; del buf990  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf1027, convolution_49, unsqueeze_2122, buf1030, squeeze_148, buf1028, primals_149, buf1032, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf1027
        del convolution_49
        del primals_149
        del squeeze_148
        del unsqueeze_2122
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1033 = aten.convolution_backward(buf1032, relu_45, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_148
        buf1034 = buf1033[0]
        buf1035 = buf1033[1]
        del buf1033
        buf1036 = buf1034; del buf1034  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_61.run(buf1036, relu_45, relu_50, buf951, buf992, 8192, 196, grid=grid(8192, 196), stream=stream0)
        del relu_45
        del relu_50
        buf1037 = buf996; del buf996  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_57.run(buf1036, buf1037, 1024, 1568, grid=grid(1024), stream=stream0)
        buf1038 = buf995; del buf995  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_58.run(buf1036, convolution_48, unsqueeze_2134, buf1038, 13312, 121, grid=grid(13312), stream=stream0)
        buf1039 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1040 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf1038, squeeze_145, buf1039, buf1040, 1024, 13, grid=grid(1024), stream=stream0)
        buf1041 = reinterpret_tensor(buf992, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf992  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_59.run(buf1036, convolution_48, unsqueeze_2134, buf1039, squeeze_145, buf1037, primals_146, buf1041, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del convolution_48
        del primals_146
        del squeeze_145
        del unsqueeze_2134
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1042 = aten.convolution_backward(buf1041, cat_8, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_8
        del primals_145
        buf1043 = buf1042[0]
        buf1044 = buf1042[1]
        del buf1042
        buf1045 = buf1020; del buf1020  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_121, buf1043, buf1045, 1352, 121, grid=grid(1352), stream=stream0)
        buf1046 = buf1021; del buf1021  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf1045, buf1046, 104, 13, grid=grid(104), stream=stream0)
        buf1047 = reinterpret_tensor(buf1045, (104, 13), (1, 104), 0); del buf1045  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_121, buf1043, convolution_47, unsqueeze_2146, buf1047, 1352, 121, grid=grid(1352), stream=stream0)
        buf1048 = empty((104, ), device='cuda', dtype=torch.float32)
        buf1049 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf1047, squeeze_142, buf1048, buf1049, 104, 13, grid=grid(104), stream=stream0)
        buf1050 = reinterpret_tensor(buf1025, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1025  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_121, buf1043, convolution_47, unsqueeze_2146, buf1048, squeeze_142, buf1046, primals_143, buf1050, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_47
        del le_121
        del primals_143
        del squeeze_142
        del unsqueeze_2146
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1051 = aten.convolution_backward(buf1050, add_254, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_254
        del primals_142
        buf1052 = buf1051[0]
        buf1053 = buf1051[1]
        del buf1051
        buf1054 = buf1048; del buf1048  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(le_122, buf1043, buf1052, buf1054, 104, 1568, grid=grid(104), stream=stream0)
        buf1055 = reinterpret_tensor(buf1047, (104, 13), (13, 1), 0); del buf1047  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_45.run(le_122, buf1043, buf1052, convolution_46, unsqueeze_2158, buf1055, 1352, 121, grid=grid(1352), stream=stream0)
        buf1056 = empty((104, ), device='cuda', dtype=torch.float32)
        buf1058 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf1055, squeeze_139, buf1056, buf1058, 104, 13, grid=grid(104), stream=stream0)
        buf1057 = buf1050; del buf1050  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(le_122, buf1043, buf1052, convolution_46, unsqueeze_2158, buf1056, squeeze_139, buf1054, primals_140, buf1057, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_46
        del le_122
        del primals_140
        del squeeze_139
        del unsqueeze_2158
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1059 = aten.convolution_backward(buf1057, add_248, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_248
        del primals_139
        buf1060 = buf1059[0]
        buf1061 = buf1059[1]
        del buf1059
        buf1062 = buf1056; del buf1056  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(le_123, buf1043, buf1060, buf1062, 104, 1568, grid=grid(104), stream=stream0)
        buf1063 = buf1055; del buf1055  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_49.run(le_123, buf1043, buf1060, convolution_45, unsqueeze_2170, buf1063, 1352, 121, grid=grid(1352), stream=stream0)
        buf1064 = empty((104, ), device='cuda', dtype=torch.float32)
        buf1066 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_46.run(buf1063, squeeze_136, buf1064, buf1066, 104, 13, grid=grid(104), stream=stream0)
        buf1065 = buf1057; del buf1057  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_50.run(le_123, buf1043, buf1060, convolution_45, unsqueeze_2170, buf1064, squeeze_136, buf1062, primals_137, buf1065, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_45
        del le_123
        del primals_137
        del squeeze_136
        del unsqueeze_2170
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1067 = aten.convolution_backward(buf1065, getitem_256, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1065
        del getitem_256
        del primals_136
        buf1068 = buf1067[0]
        buf1069 = buf1067[1]
        del buf1067
        buf1070 = buf1043; del buf1043  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_51.run(buf1070, le_124, buf1068, buf1060, buf1052, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf1052
        del buf1060
        del le_124
        buf1071 = buf1030; del buf1030  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf1070, buf1071, 416, 1568, grid=grid(416), stream=stream0)
        buf1072 = buf1029; del buf1029  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf1070, convolution_44, unsqueeze_2182, buf1072, 5408, 121, grid=grid(5408), stream=stream0)
        buf1073 = empty((416, ), device='cuda', dtype=torch.float32)
        buf1074 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_54.run(buf1072, squeeze_133, buf1073, buf1074, 416, 13, grid=grid(416), stream=stream0)
        del buf1072
        buf1075 = buf1032; del buf1032  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_55.run(buf1070, convolution_44, unsqueeze_2182, buf1073, squeeze_133, buf1071, primals_134, buf1075, 1568, 416, grid=grid(1568, 416), stream=stream0)
        del buf1070
        del convolution_44
        del primals_134
        del squeeze_133
        del unsqueeze_2182
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1076 = aten.convolution_backward(buf1075, relu_40, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_133
        buf1077 = buf1076[0]
        buf1078 = buf1076[1]
        del buf1076
        buf1079 = buf1039; del buf1039  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_35.run(relu_40, buf1036, buf1077, buf1079, 1024, 1568, grid=grid(1024), stream=stream0)
        buf1080 = buf1038; del buf1038  # reuse
        buf1087 = empty((1024, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_62.run(relu_40, buf1036, buf1077, convolution_43, unsqueeze_2194, convolution_42, unsqueeze_2206, buf1080, buf1087, 13312, 121, grid=grid(13312), stream=stream0)
        buf1081 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1083 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf1080, squeeze_130, buf1081, buf1083, 1024, 13, grid=grid(1024), stream=stream0)
        del buf1080
        buf1088 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf1090 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_37.run(buf1087, squeeze_127, buf1088, buf1090, 1024, 13, grid=grid(1024), stream=stream0)
        del buf1087
        buf1082 = buf1041; del buf1041  # reuse
        buf1089 = reinterpret_tensor(buf951, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf951  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_63.run(relu_40, buf1036, buf1077, convolution_43, unsqueeze_2194, buf1081, squeeze_130, buf1079, primals_131, convolution_42, unsqueeze_2206, buf1088, squeeze_127, primals_128, buf1082, buf1089, 1568, 1024, grid=grid(1568, 1024), stream=stream0)
        del buf1036
        del buf1077
        del buf1081
        del buf1088
        del convolution_42
        del convolution_43
        del primals_128
        del primals_131
        del relu_40
        del squeeze_127
        del squeeze_130
        del unsqueeze_2194
        del unsqueeze_2206
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1084 = aten.convolution_backward(buf1082, relu_35, primals_130, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1082
        del primals_130
        buf1085 = buf1084[0]
        buf1086 = buf1084[1]
        del buf1084
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1091 = aten.convolution_backward(buf1089, cat_7, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1089
        del cat_7
        del primals_127
        buf1092 = buf1091[0]
        buf1093 = buf1091[1]
        del buf1091
        buf1094 = reinterpret_tensor(buf1075, (8, 104, 28, 28), (81536, 784, 28, 1), 0); del buf1075  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_64.run(buf1092, buf1094, 652288, grid=grid(652288), stream=stream0)
        buf1095 = buf1063; del buf1063  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(le_126, buf1092, buf1095, 1352, 121, grid=grid(1352), stream=stream0)
        buf1096 = buf1064; del buf1064  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf1095, buf1096, 104, 13, grid=grid(104), stream=stream0)
        buf1097 = reinterpret_tensor(buf1095, (104, 13), (1, 104), 0); del buf1095  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_41.run(le_126, buf1092, convolution_41, unsqueeze_2218, buf1097, 1352, 121, grid=grid(1352), stream=stream0)
        buf1098 = empty((104, ), device='cuda', dtype=torch.float32)
        buf1099 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf1097, squeeze_124, buf1098, buf1099, 104, 13, grid=grid(104), stream=stream0)
        buf1100 = reinterpret_tensor(buf1068, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf1068  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43.run(le_126, buf1092, convolution_41, unsqueeze_2218, buf1098, squeeze_124, buf1096, primals_125, buf1100, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_41
        del le_126
        del primals_125
        del squeeze_124
        del unsqueeze_2218
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1101 = aten.convolution_backward(buf1100, getitem_238, primals_124, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_238
        del primals_124
        buf1102 = buf1101[0]
        buf1103 = buf1101[1]
        del buf1101
        buf1104 = reinterpret_tensor(buf1097, (104, 13), (13, 1), 0); del buf1097  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_65.run(le_127, buf1092, buf1104, 1352, 121, grid=grid(1352), stream=stream0)
        buf1105 = buf1098; del buf1098  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf1104, buf1105, 104, 13, grid=grid(104), stream=stream0)
        buf1106 = reinterpret_tensor(buf1104, (104, 13), (1, 104), 0); del buf1104  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(le_127, buf1092, convolution_40, unsqueeze_2230, buf1106, 1352, 121, grid=grid(1352), stream=stream0)
        buf1107 = empty((104, ), device='cuda', dtype=torch.float32)
        buf1108 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf1106, squeeze_121, buf1107, buf1108, 104, 13, grid=grid(104), stream=stream0)
        buf1109 = buf1100; del buf1100  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67.run(le_127, buf1092, convolution_40, unsqueeze_2230, buf1107, squeeze_121, buf1105, primals_122, buf1109, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del convolution_40
        del le_127
        del primals_122
        del squeeze_121
        del unsqueeze_2230
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1110 = aten.convolution_backward(buf1109, getitem_231, primals_121, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_231
        del primals_121
        buf1111 = buf1110[0]
        buf1112 = buf1110[1]
        del buf1110
        buf1113 = reinterpret_tensor(buf1106, (104, 13), (13, 1), 0); del buf1106  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_68.run(le_128, buf1092, buf1113, 1352, 121, grid=grid(1352), stream=stream0)
        buf1114 = buf1107; del buf1107  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf1113, buf1114, 104, 13, grid=grid(104), stream=stream0)
        buf1115 = reinterpret_tensor(buf1113, (104, 13), (1, 104), 0); del buf1113  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_69.run(le_128, buf1092, convolution_39, unsqueeze_2242, buf1115, 1352, 121, grid=grid(1352), stream=stream0)
        buf1116 = empty((104, ), device='cuda', dtype=torch.float32)
        buf1117 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_42.run(buf1115, squeeze_118, buf1116, buf1117, 104, 13, grid=grid(104), stream=stream0)
        del buf1115
        buf1118 = buf1109; del buf1109  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70.run(le_128, buf1092, convolution_39, unsqueeze_2242, buf1116, squeeze_118, buf1114, primals_119, buf1118, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del buf1092
        del convolution_39
        del le_128
        del primals_119
        del squeeze_118
        del unsqueeze_2242
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1119 = aten.convolution_backward(buf1118, getitem_224, primals_118, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1118
        del getitem_224
        del primals_118
        buf1120 = buf1119[0]
        buf1121 = buf1119[1]
        del buf1119
        buf1122 = empty((8, 416, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_71.run(le_129, buf1120, buf1111, buf1102, buf1094, buf1122, 3328, 784, grid=grid(3328, 784), stream=stream0)
        del buf1094
        del buf1102
        del buf1111
        del le_129
        buf1123 = buf1073; del buf1073  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_72.run(buf1122, buf1123, 416, 6272, grid=grid(416), stream=stream0)
        buf1124 = empty((416, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_73.run(buf1122, convolution_38, unsqueeze_2254, buf1124, 20384, 128, grid=grid(20384), stream=stream0)
        buf1125 = empty((416, ), device='cuda', dtype=torch.float32)
        buf1126 = empty((416, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_74.run(buf1124, squeeze_115, buf1125, buf1126, 416, 49, grid=grid(416), stream=stream0)
        buf1127 = empty_strided((8, 416, 28, 28), (326144, 1, 11648, 416), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_75.run(buf1122, convolution_38, unsqueeze_2254, buf1125, squeeze_115, buf1123, primals_116, buf1127, 6272, 416, grid=grid(6272, 416), stream=stream0)
        del buf1122
        del convolution_38
        del primals_116
        del squeeze_115
        del unsqueeze_2254
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1128 = aten.convolution_backward(buf1127, relu_35, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_115
        buf1129 = buf1128[0]
        buf1130 = buf1128[1]
        del buf1128
        buf1131 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_76.run(relu_35, buf1085, buf1129, buf1131, 512, 6272, grid=grid(512), stream=stream0)
        buf1132 = empty((512, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_77.run(relu_35, buf1085, buf1129, convolution_37, unsqueeze_2266, buf1132, 25088, 128, grid=grid(25088), stream=stream0)
        buf1133 = empty((512, ), device='cuda', dtype=torch.float32)
        buf1135 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_78.run(buf1132, squeeze_112, buf1133, buf1135, 512, 49, grid=grid(512), stream=stream0)
        buf1134 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_79.run(relu_35, buf1085, buf1129, convolution_37, unsqueeze_2266, buf1133, squeeze_112, buf1131, primals_113, buf1134, 6272, 512, grid=grid(6272, 512), stream=stream0)
        del convolution_37
        del primals_113
        del squeeze_112
        del unsqueeze_2266
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1136 = aten.convolution_backward(buf1134, cat_6, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1134
        del cat_6
        del primals_112
        buf1137 = buf1136[0]
        buf1138 = buf1136[1]
        del buf1136
        buf1139 = empty((52, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(le_131, buf1137, buf1139, 2548, 128, grid=grid(2548), stream=stream0)
        buf1140 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_81.run(buf1139, buf1140, 52, 49, grid=grid(52), stream=stream0)
        buf1141 = reinterpret_tensor(buf1139, (52, 49), (1, 52), 0); del buf1139  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_82.run(le_131, buf1137, convolution_36, unsqueeze_2278, buf1141, 2548, 128, grid=grid(2548), stream=stream0)
        buf1142 = empty((52, ), device='cuda', dtype=torch.float32)
        buf1143 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_83.run(buf1141, squeeze_109, buf1142, buf1143, 52, 49, grid=grid(52), stream=stream0)
        buf1144 = reinterpret_tensor(buf133, (8, 52, 28, 28), (40768, 1, 1456, 52), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_84.run(le_131, buf1137, convolution_36, unsqueeze_2278, buf1142, squeeze_109, buf1140, primals_110, buf1144, 6272, 52, grid=grid(6272, 52), stream=stream0)
        del convolution_36
        del le_131
        del primals_110
        del squeeze_109
        del unsqueeze_2278
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1145 = aten.convolution_backward(buf1144, add_195, primals_109, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_195
        del primals_109
        buf1146 = buf1145[0]
        buf1147 = buf1145[1]
        del buf1145
        buf1148 = buf1142; del buf1142  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_85.run(le_132, buf1137, buf1146, buf1148, 52, 6272, grid=grid(52), stream=stream0)
        buf1149 = reinterpret_tensor(buf1141, (52, 49), (49, 1), 0); del buf1141  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_86.run(le_132, buf1137, buf1146, convolution_35, unsqueeze_2290, buf1149, 2548, 128, grid=grid(2548), stream=stream0)
        buf1150 = empty((52, ), device='cuda', dtype=torch.float32)
        buf1152 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_87.run(buf1149, squeeze_106, buf1150, buf1152, 52, 49, grid=grid(52), stream=stream0)
        buf1151 = buf1144; del buf1144  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_88.run(le_132, buf1137, buf1146, convolution_35, unsqueeze_2290, buf1150, squeeze_106, buf1148, primals_107, buf1151, 6272, 52, grid=grid(6272, 52), stream=stream0)
        del convolution_35
        del le_132
        del primals_107
        del squeeze_106
        del unsqueeze_2290
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1153 = aten.convolution_backward(buf1151, add_189, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_189
        del primals_106
        buf1154 = buf1153[0]
        buf1155 = buf1153[1]
        del buf1153
        buf1156 = buf1150; del buf1150  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_89.run(le_133, buf1137, buf1154, buf1156, 52, 6272, grid=grid(52), stream=stream0)
        buf1157 = buf1149; del buf1149  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_90.run(le_133, buf1137, buf1154, convolution_34, unsqueeze_2302, buf1157, 2548, 128, grid=grid(2548), stream=stream0)
        buf1158 = empty((52, ), device='cuda', dtype=torch.float32)
        buf1160 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_87.run(buf1157, squeeze_103, buf1158, buf1160, 52, 49, grid=grid(52), stream=stream0)
        buf1159 = buf1151; del buf1151  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_91.run(le_133, buf1137, buf1154, convolution_34, unsqueeze_2302, buf1158, squeeze_103, buf1156, primals_104, buf1159, 6272, 52, grid=grid(6272, 52), stream=stream0)
        del convolution_34
        del le_133
        del primals_104
        del squeeze_103
        del unsqueeze_2302
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1161 = aten.convolution_backward(buf1159, getitem_194, primals_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1159
        del getitem_194
        del primals_103
        buf1162 = buf1161[0]
        buf1163 = buf1161[1]
        del buf1161
        buf1164 = buf1137; del buf1137  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_92.run(buf1164, le_134, buf1162, buf1154, buf1146, 1664, 784, grid=grid(1664, 784), stream=stream0)
        del buf1146
        del buf1154
        del le_134
        buf1165 = buf129; del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_93.run(buf1164, buf1165, 208, 6272, grid=grid(208), stream=stream0)
        buf1166 = empty((208, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_94.run(buf1164, convolution_33, unsqueeze_2314, buf1166, 10192, 128, grid=grid(10192), stream=stream0)
        buf1167 = empty((208, ), device='cuda', dtype=torch.float32)
        buf1168 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_95.run(buf1166, squeeze_100, buf1167, buf1168, 208, 49, grid=grid(208), stream=stream0)
        buf1169 = reinterpret_tensor(buf140, (8, 208, 28, 28), (163072, 1, 5824, 208), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_96.run(buf1164, convolution_33, unsqueeze_2314, buf1167, squeeze_100, buf1165, primals_101, buf1169, 6272, 208, grid=grid(6272, 208), stream=stream0)
        del buf1164
        del convolution_33
        del primals_101
        del squeeze_100
        del unsqueeze_2314
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1170 = aten.convolution_backward(buf1169, relu_30, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_100
        buf1171 = buf1170[0]
        buf1172 = buf1170[1]
        del buf1170
        buf1173 = buf1085; del buf1085  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_97.run(buf1173, relu_30, relu_35, buf1129, buf1171, 4096, 784, grid=grid(4096, 784), stream=stream0)
        del buf1129
        del relu_30
        del relu_35
        buf1174 = buf1133; del buf1133  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_98.run(buf1173, buf1174, 512, 6272, grid=grid(512), stream=stream0)
        buf1175 = buf1132; del buf1132  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_99.run(buf1173, convolution_32, unsqueeze_2326, buf1175, 25088, 128, grid=grid(25088), stream=stream0)
        buf1176 = empty((512, ), device='cuda', dtype=torch.float32)
        buf1177 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_78.run(buf1175, squeeze_97, buf1176, buf1177, 512, 49, grid=grid(512), stream=stream0)
        buf1178 = reinterpret_tensor(buf1171, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf1171  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_100.run(buf1173, convolution_32, unsqueeze_2326, buf1176, squeeze_97, buf1174, primals_98, buf1178, 6272, 512, grid=grid(6272, 512), stream=stream0)
        del convolution_32
        del primals_98
        del squeeze_97
        del unsqueeze_2326
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1179 = aten.convolution_backward(buf1178, cat_5, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_5
        del primals_97
        buf1180 = buf1179[0]
        buf1181 = buf1179[1]
        del buf1179
        buf1182 = buf1157; del buf1157  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(le_136, buf1180, buf1182, 2548, 128, grid=grid(2548), stream=stream0)
        buf1183 = buf1158; del buf1158  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_81.run(buf1182, buf1183, 52, 49, grid=grid(52), stream=stream0)
        buf1184 = reinterpret_tensor(buf1182, (52, 49), (1, 52), 0); del buf1182  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_82.run(le_136, buf1180, convolution_31, unsqueeze_2338, buf1184, 2548, 128, grid=grid(2548), stream=stream0)
        buf1185 = empty((52, ), device='cuda', dtype=torch.float32)
        buf1186 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_83.run(buf1184, squeeze_94, buf1185, buf1186, 52, 49, grid=grid(52), stream=stream0)
        buf1187 = reinterpret_tensor(buf1162, (8, 52, 28, 28), (40768, 1, 1456, 52), 0); del buf1162  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_84.run(le_136, buf1180, convolution_31, unsqueeze_2338, buf1185, squeeze_94, buf1183, primals_95, buf1187, 6272, 52, grid=grid(6272, 52), stream=stream0)
        del convolution_31
        del le_136
        del primals_95
        del squeeze_94
        del unsqueeze_2338
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1188 = aten.convolution_backward(buf1187, add_167, primals_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_167
        del primals_94
        buf1189 = buf1188[0]
        buf1190 = buf1188[1]
        del buf1188
        buf1191 = buf1185; del buf1185  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_85.run(le_137, buf1180, buf1189, buf1191, 52, 6272, grid=grid(52), stream=stream0)
        buf1192 = reinterpret_tensor(buf1184, (52, 49), (49, 1), 0); del buf1184  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_86.run(le_137, buf1180, buf1189, convolution_30, unsqueeze_2350, buf1192, 2548, 128, grid=grid(2548), stream=stream0)
        buf1193 = empty((52, ), device='cuda', dtype=torch.float32)
        buf1195 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_87.run(buf1192, squeeze_91, buf1193, buf1195, 52, 49, grid=grid(52), stream=stream0)
        buf1194 = buf1187; del buf1187  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_88.run(le_137, buf1180, buf1189, convolution_30, unsqueeze_2350, buf1193, squeeze_91, buf1191, primals_92, buf1194, 6272, 52, grid=grid(6272, 52), stream=stream0)
        del convolution_30
        del le_137
        del primals_92
        del squeeze_91
        del unsqueeze_2350
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1196 = aten.convolution_backward(buf1194, add_161, primals_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_161
        del primals_91
        buf1197 = buf1196[0]
        buf1198 = buf1196[1]
        del buf1196
        buf1199 = buf1193; del buf1193  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_89.run(le_138, buf1180, buf1197, buf1199, 52, 6272, grid=grid(52), stream=stream0)
        buf1200 = buf1192; del buf1192  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_90.run(le_138, buf1180, buf1197, convolution_29, unsqueeze_2362, buf1200, 2548, 128, grid=grid(2548), stream=stream0)
        buf1201 = empty((52, ), device='cuda', dtype=torch.float32)
        buf1203 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_87.run(buf1200, squeeze_88, buf1201, buf1203, 52, 49, grid=grid(52), stream=stream0)
        buf1202 = buf1194; del buf1194  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_91.run(le_138, buf1180, buf1197, convolution_29, unsqueeze_2362, buf1201, squeeze_88, buf1199, primals_89, buf1202, 6272, 52, grid=grid(6272, 52), stream=stream0)
        del convolution_29
        del le_138
        del primals_89
        del squeeze_88
        del unsqueeze_2362
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1204 = aten.convolution_backward(buf1202, getitem_164, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1202
        del getitem_164
        del primals_88
        buf1205 = buf1204[0]
        buf1206 = buf1204[1]
        del buf1204
        buf1207 = buf1180; del buf1180  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_92.run(buf1207, le_139, buf1205, buf1197, buf1189, 1664, 784, grid=grid(1664, 784), stream=stream0)
        del buf1189
        del buf1197
        del le_139
        buf1208 = buf1167; del buf1167  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_93.run(buf1207, buf1208, 208, 6272, grid=grid(208), stream=stream0)
        buf1209 = buf1166; del buf1166  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_94.run(buf1207, convolution_28, unsqueeze_2374, buf1209, 10192, 128, grid=grid(10192), stream=stream0)
        buf1210 = empty((208, ), device='cuda', dtype=torch.float32)
        buf1211 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_95.run(buf1209, squeeze_85, buf1210, buf1211, 208, 49, grid=grid(208), stream=stream0)
        buf1212 = buf1169; del buf1169  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_96.run(buf1207, convolution_28, unsqueeze_2374, buf1210, squeeze_85, buf1208, primals_86, buf1212, 6272, 208, grid=grid(6272, 208), stream=stream0)
        del buf1207
        del convolution_28
        del primals_86
        del squeeze_85
        del unsqueeze_2374
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1213 = aten.convolution_backward(buf1212, relu_25, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_85
        buf1214 = buf1213[0]
        buf1215 = buf1213[1]
        del buf1213
        buf1216 = buf1176; del buf1176  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_76.run(relu_25, buf1173, buf1214, buf1216, 512, 6272, grid=grid(512), stream=stream0)
        buf1217 = buf1175; del buf1175  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_77.run(relu_25, buf1173, buf1214, convolution_27, unsqueeze_2386, buf1217, 25088, 128, grid=grid(25088), stream=stream0)
        buf1218 = empty((512, ), device='cuda', dtype=torch.float32)
        buf1220 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_78.run(buf1217, squeeze_82, buf1218, buf1220, 512, 49, grid=grid(512), stream=stream0)
        buf1219 = buf1178; del buf1178  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_79.run(relu_25, buf1173, buf1214, convolution_27, unsqueeze_2386, buf1218, squeeze_82, buf1216, primals_83, buf1219, 6272, 512, grid=grid(6272, 512), stream=stream0)
        del convolution_27
        del primals_83
        del squeeze_82
        del unsqueeze_2386
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1221 = aten.convolution_backward(buf1219, cat_4, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1219
        del cat_4
        del primals_82
        buf1222 = buf1221[0]
        buf1223 = buf1221[1]
        del buf1221
        buf1224 = buf1200; del buf1200  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(le_141, buf1222, buf1224, 2548, 128, grid=grid(2548), stream=stream0)
        buf1225 = buf1201; del buf1201  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_81.run(buf1224, buf1225, 52, 49, grid=grid(52), stream=stream0)
        buf1226 = reinterpret_tensor(buf1224, (52, 49), (1, 52), 0); del buf1224  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_82.run(le_141, buf1222, convolution_26, unsqueeze_2398, buf1226, 2548, 128, grid=grid(2548), stream=stream0)
        buf1227 = empty((52, ), device='cuda', dtype=torch.float32)
        buf1228 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_83.run(buf1226, squeeze_79, buf1227, buf1228, 52, 49, grid=grid(52), stream=stream0)
        buf1229 = reinterpret_tensor(buf1205, (8, 52, 28, 28), (40768, 1, 1456, 52), 0); del buf1205  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_84.run(le_141, buf1222, convolution_26, unsqueeze_2398, buf1227, squeeze_79, buf1225, primals_80, buf1229, 6272, 52, grid=grid(6272, 52), stream=stream0)
        del convolution_26
        del le_141
        del primals_80
        del squeeze_79
        del unsqueeze_2398
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1230 = aten.convolution_backward(buf1229, add_139, primals_79, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_139
        del primals_79
        buf1231 = buf1230[0]
        buf1232 = buf1230[1]
        del buf1230
        buf1233 = buf1227; del buf1227  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_85.run(le_142, buf1222, buf1231, buf1233, 52, 6272, grid=grid(52), stream=stream0)
        buf1234 = reinterpret_tensor(buf1226, (52, 49), (49, 1), 0); del buf1226  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_86.run(le_142, buf1222, buf1231, convolution_25, unsqueeze_2410, buf1234, 2548, 128, grid=grid(2548), stream=stream0)
        buf1235 = empty((52, ), device='cuda', dtype=torch.float32)
        buf1237 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_87.run(buf1234, squeeze_76, buf1235, buf1237, 52, 49, grid=grid(52), stream=stream0)
        buf1236 = buf1229; del buf1229  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_88.run(le_142, buf1222, buf1231, convolution_25, unsqueeze_2410, buf1235, squeeze_76, buf1233, primals_77, buf1236, 6272, 52, grid=grid(6272, 52), stream=stream0)
        del convolution_25
        del le_142
        del primals_77
        del squeeze_76
        del unsqueeze_2410
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1238 = aten.convolution_backward(buf1236, add_133, primals_76, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_133
        del primals_76
        buf1239 = buf1238[0]
        buf1240 = buf1238[1]
        del buf1238
        buf1241 = buf1235; del buf1235  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_89.run(le_143, buf1222, buf1239, buf1241, 52, 6272, grid=grid(52), stream=stream0)
        buf1242 = buf1234; del buf1234  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_90.run(le_143, buf1222, buf1239, convolution_24, unsqueeze_2422, buf1242, 2548, 128, grid=grid(2548), stream=stream0)
        buf1243 = empty((52, ), device='cuda', dtype=torch.float32)
        buf1245 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_87.run(buf1242, squeeze_73, buf1243, buf1245, 52, 49, grid=grid(52), stream=stream0)
        buf1244 = buf1236; del buf1236  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_91.run(le_143, buf1222, buf1239, convolution_24, unsqueeze_2422, buf1243, squeeze_73, buf1241, primals_74, buf1244, 6272, 52, grid=grid(6272, 52), stream=stream0)
        del convolution_24
        del le_143
        del primals_74
        del squeeze_73
        del unsqueeze_2422
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1246 = aten.convolution_backward(buf1244, getitem_134, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1244
        del getitem_134
        del primals_73
        buf1247 = buf1246[0]
        buf1248 = buf1246[1]
        del buf1246
        buf1249 = buf1222; del buf1222  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_92.run(buf1249, le_144, buf1247, buf1239, buf1231, 1664, 784, grid=grid(1664, 784), stream=stream0)
        del buf1231
        del buf1239
        del le_144
        buf1250 = buf1210; del buf1210  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_93.run(buf1249, buf1250, 208, 6272, grid=grid(208), stream=stream0)
        buf1251 = buf1209; del buf1209  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_94.run(buf1249, convolution_23, unsqueeze_2434, buf1251, 10192, 128, grid=grid(10192), stream=stream0)
        buf1252 = empty((208, ), device='cuda', dtype=torch.float32)
        buf1253 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_95.run(buf1251, squeeze_70, buf1252, buf1253, 208, 49, grid=grid(208), stream=stream0)
        del buf1251
        buf1254 = buf1212; del buf1212  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_96.run(buf1249, convolution_23, unsqueeze_2434, buf1252, squeeze_70, buf1250, primals_71, buf1254, 6272, 208, grid=grid(6272, 208), stream=stream0)
        del buf1249
        del convolution_23
        del primals_71
        del squeeze_70
        del unsqueeze_2434
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1255 = aten.convolution_backward(buf1254, relu_20, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_70
        buf1256 = buf1255[0]
        buf1257 = buf1255[1]
        del buf1255
        buf1258 = buf1173; del buf1173  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_97.run(buf1258, relu_20, relu_25, buf1214, buf1256, 4096, 784, grid=grid(4096, 784), stream=stream0)
        del relu_20
        del relu_25
        buf1259 = buf1218; del buf1218  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_98.run(buf1258, buf1259, 512, 6272, grid=grid(512), stream=stream0)
        buf1260 = buf1217; del buf1217  # reuse
        buf1267 = empty((512, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_101.run(buf1258, convolution_22, unsqueeze_2446, convolution_21, unsqueeze_2458, buf1260, buf1267, 25088, 128, grid=grid(25088), stream=stream0)
        buf1261 = empty((512, ), device='cuda', dtype=torch.float32)
        buf1262 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_78.run(buf1260, squeeze_67, buf1261, buf1262, 512, 49, grid=grid(512), stream=stream0)
        del buf1260
        buf1268 = empty((512, ), device='cuda', dtype=torch.float32)
        buf1269 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_78.run(buf1267, squeeze_64, buf1268, buf1269, 512, 49, grid=grid(512), stream=stream0)
        del buf1267
        buf1263 = reinterpret_tensor(buf1256, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf1256  # reuse
        buf1270 = reinterpret_tensor(buf1214, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf1214  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_102.run(buf1258, convolution_22, unsqueeze_2446, buf1261, squeeze_67, buf1259, primals_68, convolution_21, unsqueeze_2458, buf1268, squeeze_64, primals_65, buf1263, buf1270, 6272, 512, grid=grid(6272, 512), stream=stream0)
        del buf1258
        del buf1261
        del buf1268
        del convolution_21
        del convolution_22
        del primals_65
        del primals_68
        del squeeze_64
        del squeeze_67
        del unsqueeze_2446
        del unsqueeze_2458
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1264 = aten.convolution_backward(buf1263, relu_15, primals_67, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1263
        del primals_67
        buf1265 = buf1264[0]
        buf1266 = buf1264[1]
        del buf1264
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1271 = aten.convolution_backward(buf1270, cat_3, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1270
        del cat_3
        del primals_64
        buf1272 = buf1271[0]
        buf1273 = buf1271[1]
        del buf1271
        buf1274 = reinterpret_tensor(buf1254, (8, 52, 56, 56), (163072, 3136, 56, 1), 0); del buf1254  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_103.run(buf1272, buf1274, 1304576, grid=grid(1304576), stream=stream0)
        buf1275 = buf1242; del buf1242  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_80.run(le_146, buf1272, buf1275, 2548, 128, grid=grid(2548), stream=stream0)
        buf1276 = buf1243; del buf1243  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_81.run(buf1275, buf1276, 52, 49, grid=grid(52), stream=stream0)
        buf1277 = reinterpret_tensor(buf1275, (52, 49), (1, 52), 0); del buf1275  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_82.run(le_146, buf1272, convolution_20, unsqueeze_2470, buf1277, 2548, 128, grid=grid(2548), stream=stream0)
        buf1278 = empty((52, ), device='cuda', dtype=torch.float32)
        buf1279 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_83.run(buf1277, squeeze_61, buf1278, buf1279, 52, 49, grid=grid(52), stream=stream0)
        buf1280 = reinterpret_tensor(buf1247, (8, 52, 28, 28), (40768, 1, 1456, 52), 0); del buf1247  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_84.run(le_146, buf1272, convolution_20, unsqueeze_2470, buf1278, squeeze_61, buf1276, primals_62, buf1280, 6272, 52, grid=grid(6272, 52), stream=stream0)
        del convolution_20
        del le_146
        del primals_62
        del squeeze_61
        del unsqueeze_2470
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1281 = aten.convolution_backward(buf1280, getitem_116, primals_61, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_116
        del primals_61
        buf1282 = buf1281[0]
        buf1283 = buf1281[1]
        del buf1281
        buf1284 = reinterpret_tensor(buf1277, (52, 49), (49, 1), 0); del buf1277  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_104.run(le_147, buf1272, buf1284, 2548, 128, grid=grid(2548), stream=stream0)
        buf1285 = buf1278; del buf1278  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_81.run(buf1284, buf1285, 52, 49, grid=grid(52), stream=stream0)
        buf1286 = reinterpret_tensor(buf1284, (52, 49), (1, 52), 0); del buf1284  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_105.run(le_147, buf1272, convolution_19, unsqueeze_2482, buf1286, 2548, 128, grid=grid(2548), stream=stream0)
        buf1287 = empty((52, ), device='cuda', dtype=torch.float32)
        buf1288 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_83.run(buf1286, squeeze_58, buf1287, buf1288, 52, 49, grid=grid(52), stream=stream0)
        buf1289 = buf1280; del buf1280  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_106.run(le_147, buf1272, convolution_19, unsqueeze_2482, buf1287, squeeze_58, buf1285, primals_59, buf1289, 6272, 52, grid=grid(6272, 52), stream=stream0)
        del convolution_19
        del le_147
        del primals_59
        del squeeze_58
        del unsqueeze_2482
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1290 = aten.convolution_backward(buf1289, getitem_109, primals_58, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_109
        del primals_58
        buf1291 = buf1290[0]
        buf1292 = buf1290[1]
        del buf1290
        buf1293 = reinterpret_tensor(buf1286, (52, 49), (49, 1), 0); del buf1286  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_107.run(le_148, buf1272, buf1293, 2548, 128, grid=grid(2548), stream=stream0)
        buf1294 = buf1287; del buf1287  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_81.run(buf1293, buf1294, 52, 49, grid=grid(52), stream=stream0)
        buf1295 = reinterpret_tensor(buf1293, (52, 49), (1, 52), 0); del buf1293  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_108.run(le_148, buf1272, convolution_18, unsqueeze_2494, buf1295, 2548, 128, grid=grid(2548), stream=stream0)
        buf1296 = empty((52, ), device='cuda', dtype=torch.float32)
        buf1297 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_83.run(buf1295, squeeze_55, buf1296, buf1297, 52, 49, grid=grid(52), stream=stream0)
        del buf1295
        buf1298 = buf1289; del buf1289  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_109.run(le_148, buf1272, convolution_18, unsqueeze_2494, buf1296, squeeze_55, buf1294, primals_56, buf1298, 6272, 52, grid=grid(6272, 52), stream=stream0)
        del buf1272
        del buf1296
        del convolution_18
        del le_148
        del primals_56
        del squeeze_55
        del unsqueeze_2494
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1299 = aten.convolution_backward(buf1298, getitem_102, primals_55, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1298
        del getitem_102
        del primals_55
        buf1300 = buf1299[0]
        buf1301 = buf1299[1]
        del buf1299
        buf1302 = empty((8, 208, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_110.run(le_149, buf1300, buf1291, buf1282, buf1274, buf1302, 1664, 3136, grid=grid(1664, 3136), stream=stream0)
        del buf1274
        del buf1282
        del buf1291
        del buf1300
        del le_149
        buf1303 = reinterpret_tensor(buf138, (208, 4), (1, 208), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_111.run(buf1302, buf1303, 832, 6272, grid=grid(832), stream=stream0)
        buf1304 = buf1252; del buf1252  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_112.run(buf1303, buf1304, 208, 4, grid=grid(208), stream=stream0)
        del buf1303
        buf1305 = empty((208, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_113.run(buf1302, convolution_17, unsqueeze_2506, buf1305, 40768, 128, grid=grid(40768), stream=stream0)
        buf1306 = empty((208, ), device='cuda', dtype=torch.float32)
        buf1307 = empty((208, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_114.run(buf1305, squeeze_52, buf1306, buf1307, 208, 196, grid=grid(208), stream=stream0)
        del buf1305
        buf1308 = empty_strided((8, 208, 56, 56), (652288, 1, 11648, 208), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_115.run(buf1302, convolution_17, unsqueeze_2506, buf1306, squeeze_52, buf1304, primals_53, buf1308, 25088, 208, grid=grid(25088, 208), stream=stream0)
        del buf1302
        del buf1306
        del convolution_17
        del primals_53
        del squeeze_52
        del unsqueeze_2506
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1309 = aten.convolution_backward(buf1308, relu_15, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1308
        del primals_52
        buf1310 = buf1309[0]
        buf1311 = buf1309[1]
        del buf1309
        buf1312 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_116.run(relu_15, buf1265, buf1310, buf1312, 256, 25088, grid=grid(256), stream=stream0)
        buf1313 = empty((256, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_117.run(relu_15, buf1265, buf1310, convolution_16, unsqueeze_2518, buf1313, 50176, 128, grid=grid(50176), stream=stream0)
        buf1314 = empty((256, ), device='cuda', dtype=torch.float32)
        buf1316 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_118.run(buf1313, squeeze_49, buf1314, buf1316, 256, 196, grid=grid(256), stream=stream0)
        buf1315 = empty_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_119.run(relu_15, buf1265, buf1310, convolution_16, unsqueeze_2518, buf1314, squeeze_49, buf1312, primals_50, buf1315, 25088, 256, grid=grid(25088, 256), stream=stream0)
        del convolution_16
        del primals_50
        del squeeze_49
        del unsqueeze_2518
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1317 = aten.convolution_backward(buf1315, cat_2, primals_49, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1315
        del cat_2
        del primals_49
        buf1318 = buf1317[0]
        buf1319 = buf1317[1]
        del buf1317
        buf1320 = empty((26, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_120.run(le_151, buf1318, buf1320, 5096, 128, grid=grid(5096), stream=stream0)
        buf1321 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_121.run(buf1320, buf1321, 26, 196, grid=grid(26), stream=stream0)
        buf1322 = reinterpret_tensor(buf1320, (26, 196), (1, 26), 0); del buf1320  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_122.run(le_151, buf1318, convolution_15, unsqueeze_2530, buf1322, 5096, 128, grid=grid(5096), stream=stream0)
        buf1323 = empty((26, ), device='cuda', dtype=torch.float32)
        buf1324 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_123.run(buf1322, squeeze_46, buf1323, buf1324, 26, 196, grid=grid(26), stream=stream0)
        buf1325 = reinterpret_tensor(buf1120, (8, 26, 56, 56), (81536, 1, 1456, 26), 0); del buf1120  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_124.run(le_151, buf1318, convolution_15, unsqueeze_2530, buf1323, squeeze_46, buf1321, primals_47, buf1325, 25088, 26, grid=grid(25088, 26), stream=stream0)
        del convolution_15
        del le_151
        del primals_47
        del squeeze_46
        del unsqueeze_2530
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1326 = aten.convolution_backward(buf1325, add_80, primals_46, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_80
        del primals_46
        buf1327 = buf1326[0]
        buf1328 = buf1326[1]
        del buf1326
        buf1329 = reinterpret_tensor(buf1116, (26, 4), (1, 26), 0); del buf1116  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_125.run(le_152, buf1318, buf1327, buf1329, 104, 6272, grid=grid(104), stream=stream0)
        buf1330 = buf1323; del buf1323  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_126.run(buf1329, buf1330, 26, 4, grid=grid(26), stream=stream0)
        buf1331 = reinterpret_tensor(buf1322, (26, 196), (196, 1), 0); del buf1322  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_127.run(le_152, buf1318, buf1327, convolution_14, unsqueeze_2542, buf1331, 5096, 128, grid=grid(5096), stream=stream0)
        buf1332 = empty((26, ), device='cuda', dtype=torch.float32)
        buf1334 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_128.run(buf1331, squeeze_43, buf1332, buf1334, 26, 196, grid=grid(26), stream=stream0)
        buf1333 = buf1325; del buf1325  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_129.run(le_152, buf1318, buf1327, convolution_14, unsqueeze_2542, buf1332, squeeze_43, buf1330, primals_44, buf1333, 25088, 26, grid=grid(25088, 26), stream=stream0)
        del convolution_14
        del le_152
        del primals_44
        del squeeze_43
        del unsqueeze_2542
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1335 = aten.convolution_backward(buf1333, add_74, primals_43, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_74
        del primals_43
        buf1336 = buf1335[0]
        buf1337 = buf1335[1]
        del buf1335
        buf1338 = buf1329; del buf1329  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_130.run(le_153, buf1318, buf1336, buf1338, 104, 6272, grid=grid(104), stream=stream0)
        buf1339 = buf1332; del buf1332  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_126.run(buf1338, buf1339, 26, 4, grid=grid(26), stream=stream0)
        buf1340 = buf1331; del buf1331  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_131.run(le_153, buf1318, buf1336, convolution_13, unsqueeze_2554, buf1340, 5096, 128, grid=grid(5096), stream=stream0)
        buf1341 = empty((26, ), device='cuda', dtype=torch.float32)
        buf1343 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_128.run(buf1340, squeeze_40, buf1341, buf1343, 26, 196, grid=grid(26), stream=stream0)
        buf1342 = buf1333; del buf1333  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_132.run(le_153, buf1318, buf1336, convolution_13, unsqueeze_2554, buf1341, squeeze_40, buf1339, primals_41, buf1342, 25088, 26, grid=grid(25088, 26), stream=stream0)
        del convolution_13
        del le_153
        del primals_41
        del squeeze_40
        del unsqueeze_2554
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1344 = aten.convolution_backward(buf1342, getitem_72, primals_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1342
        del getitem_72
        del primals_40
        buf1345 = buf1344[0]
        buf1346 = buf1344[1]
        del buf1344
        buf1347 = buf1318; del buf1318  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_133.run(buf1347, le_154, buf1345, buf1336, buf1327, 832, 3136, grid=grid(832, 3136), stream=stream0)
        del buf1327
        del buf1336
        del le_154
        buf1348 = reinterpret_tensor(buf1125, (104, 4), (1, 104), 0); del buf1125  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_134.run(buf1347, buf1348, 416, 6272, grid=grid(416), stream=stream0)
        buf1349 = reinterpret_tensor(buf1338, (104, ), (1, ), 0); del buf1338  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_135.run(buf1348, buf1349, 104, 4, grid=grid(104), stream=stream0)
        buf1350 = reinterpret_tensor(buf1124, (104, 196), (196, 1), 0); del buf1124  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_136.run(buf1347, convolution_12, unsqueeze_2566, buf1350, 20384, 128, grid=grid(20384), stream=stream0)
        buf1351 = empty((104, ), device='cuda', dtype=torch.float32)
        buf1352 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_137.run(buf1350, squeeze_37, buf1351, buf1352, 104, 196, grid=grid(104), stream=stream0)
        buf1353 = reinterpret_tensor(buf1127, (8, 104, 56, 56), (326144, 1, 5824, 104), 0); del buf1127  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_138.run(buf1347, convolution_12, unsqueeze_2566, buf1351, squeeze_37, buf1349, primals_38, buf1353, 25088, 104, grid=grid(25088, 104), stream=stream0)
        del buf1347
        del convolution_12
        del primals_38
        del squeeze_37
        del unsqueeze_2566
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1354 = aten.convolution_backward(buf1353, relu_10, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_37
        buf1355 = buf1354[0]
        buf1356 = buf1354[1]
        del buf1354
        buf1357 = buf1265; del buf1265  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_139.run(buf1357, relu_10, relu_15, buf1310, buf1355, 2048, 3136, grid=grid(2048, 3136), stream=stream0)
        del relu_10
        del relu_15
        buf1358 = buf1314; del buf1314  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_140.run(buf1357, buf1358, 256, 25088, grid=grid(256), stream=stream0)
        buf1359 = buf1313; del buf1313  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_141.run(buf1357, convolution_11, unsqueeze_2578, buf1359, 50176, 128, grid=grid(50176), stream=stream0)
        buf1360 = empty((256, ), device='cuda', dtype=torch.float32)
        buf1361 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_118.run(buf1359, squeeze_34, buf1360, buf1361, 256, 196, grid=grid(256), stream=stream0)
        buf1362 = reinterpret_tensor(buf1355, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf1355  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_142.run(buf1357, convolution_11, unsqueeze_2578, buf1360, squeeze_34, buf1358, primals_35, buf1362, 25088, 256, grid=grid(25088, 256), stream=stream0)
        del convolution_11
        del primals_35
        del squeeze_34
        del unsqueeze_2578
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1363 = aten.convolution_backward(buf1362, cat_1, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_1
        del primals_34
        buf1364 = buf1363[0]
        buf1365 = buf1363[1]
        del buf1363
        buf1366 = buf1340; del buf1340  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_120.run(le_156, buf1364, buf1366, 5096, 128, grid=grid(5096), stream=stream0)
        buf1367 = buf1341; del buf1341  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_121.run(buf1366, buf1367, 26, 196, grid=grid(26), stream=stream0)
        buf1368 = reinterpret_tensor(buf1366, (26, 196), (1, 26), 0); del buf1366  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_122.run(le_156, buf1364, convolution_10, unsqueeze_2590, buf1368, 5096, 128, grid=grid(5096), stream=stream0)
        buf1369 = empty((26, ), device='cuda', dtype=torch.float32)
        buf1370 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_123.run(buf1368, squeeze_31, buf1369, buf1370, 26, 196, grid=grid(26), stream=stream0)
        buf1371 = reinterpret_tensor(buf1345, (8, 26, 56, 56), (81536, 1, 1456, 26), 0); del buf1345  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_124.run(le_156, buf1364, convolution_10, unsqueeze_2590, buf1369, squeeze_31, buf1367, primals_32, buf1371, 25088, 26, grid=grid(25088, 26), stream=stream0)
        del convolution_10
        del le_156
        del primals_32
        del squeeze_31
        del unsqueeze_2590
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1372 = aten.convolution_backward(buf1371, add_52, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_52
        del primals_31
        buf1373 = buf1372[0]
        buf1374 = buf1372[1]
        del buf1372
        buf1375 = reinterpret_tensor(buf1351, (26, 4), (1, 26), 0); del buf1351  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_125.run(le_157, buf1364, buf1373, buf1375, 104, 6272, grid=grid(104), stream=stream0)
        buf1376 = buf1369; del buf1369  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_126.run(buf1375, buf1376, 26, 4, grid=grid(26), stream=stream0)
        buf1377 = reinterpret_tensor(buf1368, (26, 196), (196, 1), 0); del buf1368  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_127.run(le_157, buf1364, buf1373, convolution_9, unsqueeze_2602, buf1377, 5096, 128, grid=grid(5096), stream=stream0)
        buf1378 = empty((26, ), device='cuda', dtype=torch.float32)
        buf1380 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_128.run(buf1377, squeeze_28, buf1378, buf1380, 26, 196, grid=grid(26), stream=stream0)
        buf1379 = buf1371; del buf1371  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_129.run(le_157, buf1364, buf1373, convolution_9, unsqueeze_2602, buf1378, squeeze_28, buf1376, primals_29, buf1379, 25088, 26, grid=grid(25088, 26), stream=stream0)
        del convolution_9
        del le_157
        del primals_29
        del squeeze_28
        del unsqueeze_2602
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1381 = aten.convolution_backward(buf1379, add_46, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_46
        del primals_28
        buf1382 = buf1381[0]
        buf1383 = buf1381[1]
        del buf1381
        buf1384 = buf1375; del buf1375  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_130.run(le_158, buf1364, buf1382, buf1384, 104, 6272, grid=grid(104), stream=stream0)
        buf1385 = buf1378; del buf1378  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_126.run(buf1384, buf1385, 26, 4, grid=grid(26), stream=stream0)
        buf1386 = buf1377; del buf1377  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_131.run(le_158, buf1364, buf1382, convolution_8, unsqueeze_2614, buf1386, 5096, 128, grid=grid(5096), stream=stream0)
        buf1387 = empty((26, ), device='cuda', dtype=torch.float32)
        buf1389 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_128.run(buf1386, squeeze_25, buf1387, buf1389, 26, 196, grid=grid(26), stream=stream0)
        buf1388 = buf1379; del buf1379  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_132.run(le_158, buf1364, buf1382, convolution_8, unsqueeze_2614, buf1387, squeeze_25, buf1385, primals_26, buf1388, 25088, 26, grid=grid(25088, 26), stream=stream0)
        del convolution_8
        del le_158
        del primals_26
        del squeeze_25
        del unsqueeze_2614
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1390 = aten.convolution_backward(buf1388, getitem_42, primals_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1388
        del getitem_42
        del primals_25
        buf1391 = buf1390[0]
        buf1392 = buf1390[1]
        del buf1390
        buf1393 = buf1364; del buf1364  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_133.run(buf1393, le_159, buf1391, buf1382, buf1373, 832, 3136, grid=grid(832, 3136), stream=stream0)
        del buf1373
        del le_159
        buf1394 = buf1348; del buf1348  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_134.run(buf1393, buf1394, 416, 6272, grid=grid(416), stream=stream0)
        buf1395 = reinterpret_tensor(buf1384, (104, ), (1, ), 0); del buf1384  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_135.run(buf1394, buf1395, 104, 4, grid=grid(104), stream=stream0)
        buf1396 = buf1350; del buf1350  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_136.run(buf1393, convolution_7, unsqueeze_2626, buf1396, 20384, 128, grid=grid(20384), stream=stream0)
        buf1397 = empty((104, ), device='cuda', dtype=torch.float32)
        buf1398 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_137.run(buf1396, squeeze_22, buf1397, buf1398, 104, 196, grid=grid(104), stream=stream0)
        buf1399 = buf1353; del buf1353  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_138.run(buf1393, convolution_7, unsqueeze_2626, buf1397, squeeze_22, buf1395, primals_23, buf1399, 25088, 104, grid=grid(25088, 104), stream=stream0)
        del buf1393
        del convolution_7
        del primals_23
        del squeeze_22
        del unsqueeze_2626
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1400 = aten.convolution_backward(buf1399, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_22
        buf1401 = buf1400[0]
        buf1402 = buf1400[1]
        del buf1400
        buf1403 = buf1360; del buf1360  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_116.run(relu_5, buf1357, buf1401, buf1403, 256, 25088, grid=grid(256), stream=stream0)
        buf1404 = buf1359; del buf1359  # reuse
        buf1411 = empty((256, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_143.run(relu_5, buf1357, buf1401, convolution_6, unsqueeze_2638, convolution_5, unsqueeze_2650, buf1404, buf1411, 50176, 128, grid=grid(50176), stream=stream0)
        buf1405 = empty((256, ), device='cuda', dtype=torch.float32)
        buf1407 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_118.run(buf1404, squeeze_19, buf1405, buf1407, 256, 196, grid=grid(256), stream=stream0)
        buf1412 = empty((256, ), device='cuda', dtype=torch.float32)
        buf1414 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_118.run(buf1411, squeeze_16, buf1412, buf1414, 256, 196, grid=grid(256), stream=stream0)
        buf1406 = buf1362; del buf1362  # reuse
        buf1413 = reinterpret_tensor(buf1310, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf1310  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_144.run(relu_5, buf1357, buf1401, convolution_6, unsqueeze_2638, buf1405, squeeze_19, buf1403, primals_20, convolution_5, unsqueeze_2650, buf1412, squeeze_16, primals_17, buf1406, buf1413, 25088, 256, grid=grid(25088, 256), stream=stream0)
        del buf1357
        del buf1401
        del buf1405
        del buf1412
        del convolution_5
        del convolution_6
        del primals_17
        del primals_20
        del relu_5
        del squeeze_16
        del squeeze_19
        del unsqueeze_2638
        del unsqueeze_2650
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1408 = aten.convolution_backward(buf1406, getitem_2, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1406
        del primals_19
        buf1409 = buf1408[0]
        buf1410 = buf1408[1]
        del buf1408
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1415 = aten.convolution_backward(buf1413, cat, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1413
        del cat
        del primals_16
        buf1416 = buf1415[0]
        buf1417 = buf1415[1]
        del buf1415
        buf1418 = buf1391; del buf1391  # reuse
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward]
        triton_poi_fused_avg_pool2d_backward_145.run(buf1416, buf1418, 652288, grid=grid(652288), stream=stream0)
        buf1419 = buf1386; del buf1386  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_120.run(le_161, buf1416, buf1419, 5096, 128, grid=grid(5096), stream=stream0)
        buf1420 = buf1387; del buf1387  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_121.run(buf1419, buf1420, 26, 196, grid=grid(26), stream=stream0)
        buf1421 = reinterpret_tensor(buf1419, (26, 196), (1, 26), 0); del buf1419  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_122.run(le_161, buf1416, convolution_4, unsqueeze_2662, buf1421, 5096, 128, grid=grid(5096), stream=stream0)
        buf1422 = empty((26, ), device='cuda', dtype=torch.float32)
        buf1423 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_123.run(buf1421, squeeze_13, buf1422, buf1423, 26, 196, grid=grid(26), stream=stream0)
        buf1424 = reinterpret_tensor(buf1382, (8, 26, 56, 56), (81536, 1, 1456, 26), 0); del buf1382  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_124.run(le_161, buf1416, convolution_4, unsqueeze_2662, buf1422, squeeze_13, buf1420, primals_14, buf1424, 25088, 26, grid=grid(25088, 26), stream=stream0)
        del convolution_4
        del le_161
        del primals_14
        del squeeze_13
        del unsqueeze_2662
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1425 = aten.convolution_backward(buf1424, getitem_24, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_24
        del primals_13
        buf1426 = buf1425[0]
        buf1427 = buf1425[1]
        del buf1425
        buf1428 = reinterpret_tensor(buf1421, (26, 196), (196, 1), 0); del buf1421  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_146.run(le_162, buf1416, buf1428, 5096, 128, grid=grid(5096), stream=stream0)
        buf1429 = buf1422; del buf1422  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_121.run(buf1428, buf1429, 26, 196, grid=grid(26), stream=stream0)
        buf1430 = reinterpret_tensor(buf1428, (26, 196), (1, 26), 0); del buf1428  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_147.run(le_162, buf1416, convolution_3, unsqueeze_2674, buf1430, 5096, 128, grid=grid(5096), stream=stream0)
        buf1431 = empty((26, ), device='cuda', dtype=torch.float32)
        buf1432 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_123.run(buf1430, squeeze_10, buf1431, buf1432, 26, 196, grid=grid(26), stream=stream0)
        buf1433 = buf1424; del buf1424  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_148.run(le_162, buf1416, convolution_3, unsqueeze_2674, buf1431, squeeze_10, buf1429, primals_11, buf1433, 25088, 26, grid=grid(25088, 26), stream=stream0)
        del convolution_3
        del le_162
        del primals_11
        del squeeze_10
        del unsqueeze_2674
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1434 = aten.convolution_backward(buf1433, getitem_17, primals_10, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_17
        del primals_10
        buf1435 = buf1434[0]
        buf1436 = buf1434[1]
        del buf1434
        buf1437 = reinterpret_tensor(buf1430, (26, 196), (196, 1), 0); del buf1430  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_149.run(le_163, buf1416, buf1437, 5096, 128, grid=grid(5096), stream=stream0)
        buf1438 = buf1431; del buf1431  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_121.run(buf1437, buf1438, 26, 196, grid=grid(26), stream=stream0)
        buf1439 = reinterpret_tensor(buf1437, (26, 196), (1, 26), 0); del buf1437  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_150.run(le_163, buf1416, convolution_2, unsqueeze_2686, buf1439, 5096, 128, grid=grid(5096), stream=stream0)
        buf1440 = empty((26, ), device='cuda', dtype=torch.float32)
        buf1441 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_123.run(buf1439, squeeze_7, buf1440, buf1441, 26, 196, grid=grid(26), stream=stream0)
        del buf1439
        buf1442 = buf1433; del buf1433  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_151.run(le_163, buf1416, convolution_2, unsqueeze_2686, buf1440, squeeze_7, buf1438, primals_8, buf1442, 25088, 26, grid=grid(25088, 26), stream=stream0)
        del buf1440
        del convolution_2
        del le_163
        del primals_8
        del squeeze_7
        del unsqueeze_2686
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1443 = aten.convolution_backward(buf1442, getitem_10, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1442
        del getitem_10
        del primals_7
        buf1444 = buf1443[0]
        buf1445 = buf1443[1]
        del buf1443
        buf1446 = buf1416; del buf1416  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.threshold_backward]
        triton_poi_fused_cat_threshold_backward_152.run(le_164, buf1444, buf1435, buf1426, buf1418, buf1446, 832, 3136, grid=grid(832, 3136), stream=stream0)
        del buf1418
        del buf1426
        del buf1435
        del buf1444
        del le_164
        buf1447 = buf1394; del buf1394  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_134.run(buf1446, buf1447, 416, 6272, grid=grid(416), stream=stream0)
        buf1448 = buf1397; del buf1397  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_135.run(buf1447, buf1448, 104, 4, grid=grid(104), stream=stream0)
        del buf1447
        buf1449 = buf1396; del buf1396  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_136.run(buf1446, convolution_1, unsqueeze_2698, buf1449, 20384, 128, grid=grid(20384), stream=stream0)
        buf1450 = empty((104, ), device='cuda', dtype=torch.float32)
        buf1451 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_137.run(buf1449, squeeze_4, buf1450, buf1451, 104, 196, grid=grid(104), stream=stream0)
        del buf1449
        buf1452 = buf1399; del buf1399  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_138.run(buf1446, convolution_1, unsqueeze_2698, buf1450, squeeze_4, buf1448, primals_5, buf1452, 25088, 104, grid=grid(25088, 104), stream=stream0)
        del buf1446
        del buf1450
        del convolution_1
        del primals_5
        del squeeze_4
        del unsqueeze_2698
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1453 = aten.convolution_backward(buf1452, getitem_2, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1452
        del getitem_2
        del primals_4
        buf1454 = buf1453[0]
        buf1455 = buf1453[1]
        del buf1453
        buf1456 = buf1409; del buf1409  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_153.run(buf1456, buf1454, 1605632, grid=grid(1605632), stream=stream0)
        del buf1454
        # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
        buf1457 = aten.max_pool2d_with_indices_backward(buf1456, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_3)
        del buf1456
        del getitem_3
        buf1458 = buf1457
        del buf1457
        buf1459 = reinterpret_tensor(buf1411, (64, 784), (1, 64), 0); del buf1411  # reuse
        buf1461 = reinterpret_tensor(buf1404, (64, 784), (1, 64), 0); del buf1404  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_154.run(relu, buf1458, convolution, unsqueeze_2710, buf1459, buf1461, 50176, 128, grid=grid(50176), stream=stream0)
        buf1460 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_155.run(buf1459, buf1460, 64, 784, grid=grid(64), stream=stream0)
        del buf1459
        buf1462 = empty((64, ), device='cuda', dtype=torch.float32)
        buf1463 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_156.run(buf1461, squeeze_1, buf1462, buf1463, 64, 784, grid=grid(64), stream=stream0)
        del buf1461
        buf1464 = buf1458; del buf1458  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_157.run(buf1464, relu, convolution, unsqueeze_2710, buf1462, squeeze_1, buf1460, primals_2, 6422528, grid=grid(6422528), stream=stream0)
        del buf1462
        del convolution
        del primals_2
        del relu
        del squeeze_1
        del unsqueeze_2710
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf1465 = aten.convolution_backward(buf1464, primals_1023, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf1464
        del primals_1
        del primals_1023
        buf1466 = buf1465[1]
        return (buf1466, buf1463, buf1460, buf1455, buf1451, buf1448, buf1445, buf1441, buf1438, buf1436, buf1432, buf1429, buf1427, buf1423, buf1420, buf1417, buf1414, buf1403, buf1410, buf1407, buf1403, buf1402, buf1398, buf1395, buf1392, buf1389, buf1385, buf1383, buf1380, buf1376, buf1374, buf1370, buf1367, buf1365, buf1361, buf1358, buf1356, buf1352, buf1349, buf1346, buf1343, buf1339, buf1337, buf1334, buf1330, buf1328, buf1324, buf1321, buf1319, buf1316, buf1312, buf1311, buf1307, buf1304, buf1301, buf1297, buf1294, buf1292, buf1288, buf1285, buf1283, buf1279, buf1276, buf1273, buf1269, buf1259, buf1266, buf1262, buf1259, buf1257, buf1253, buf1250, buf1248, buf1245, buf1241, buf1240, buf1237, buf1233, buf1232, buf1228, buf1225, buf1223, buf1220, buf1216, buf1215, buf1211, buf1208, buf1206, buf1203, buf1199, buf1198, buf1195, buf1191, buf1190, buf1186, buf1183, buf1181, buf1177, buf1174, buf1172, buf1168, buf1165, buf1163, buf1160, buf1156, buf1155, buf1152, buf1148, buf1147, buf1143, buf1140, buf1138, buf1135, buf1131, buf1130, buf1126, buf1123, buf1121, buf1117, buf1114, buf1112, buf1108, buf1105, buf1103, buf1099, buf1096, buf1093, buf1090, buf1079, buf1086, buf1083, buf1079, buf1078, buf1074, buf1071, buf1069, buf1066, buf1062, buf1061, buf1058, buf1054, buf1053, buf1049, buf1046, buf1044, buf1040, buf1037, buf1035, buf1031, buf1028, buf1026, buf1023, buf1019, buf1018, buf1015, buf1011, buf1010, buf1006, buf1003, buf1001, buf998, buf994, buf993, buf989, buf986, buf984, buf981, buf977, buf976, buf973, buf969, buf968, buf964, buf961, buf959, buf955, buf952, buf950, buf946, buf943, buf941, buf938, buf934, buf933, buf930, buf926, buf925, buf921, buf918, buf916, buf913, buf909, buf908, buf904, buf901, buf899, buf896, buf892, buf891, buf888, buf884, buf883, buf879, buf876, buf874, buf870, buf867, buf865, buf861, buf858, buf856, buf853, buf849, buf848, buf845, buf841, buf840, buf836, buf833, buf831, buf828, buf824, buf823, buf819, buf816, buf814, buf811, buf807, buf806, buf803, buf799, buf798, buf794, buf791, buf789, buf785, buf782, buf780, buf776, buf773, buf771, buf768, buf764, buf763, buf760, buf756, buf755, buf751, buf748, buf746, buf743, buf739, buf738, buf734, buf731, buf729, buf726, buf722, buf721, buf718, buf714, buf713, buf709, buf706, buf704, buf700, buf697, buf695, buf691, buf688, buf686, buf683, buf679, buf678, buf675, buf671, buf670, buf666, buf663, buf661, buf658, buf654, buf653, buf649, buf646, buf644, buf641, buf637, buf636, buf633, buf629, buf628, buf624, buf621, buf619, buf615, buf612, buf610, buf606, buf603, buf601, buf598, buf594, buf593, buf590, buf586, buf585, buf581, buf578, buf576, buf573, buf569, buf568, buf564, buf561, buf559, buf556, buf552, buf551, buf548, buf544, buf543, buf539, buf536, buf534, buf530, buf527, buf525, buf521, buf518, buf516, buf513, buf509, buf508, buf505, buf501, buf500, buf496, buf493, buf491, buf488, buf484, buf483, buf479, buf476, buf474, buf471, buf467, buf466, buf463, buf459, buf458, buf454, buf451, buf449, buf445, buf442, buf440, buf436, buf433, buf431, buf428, buf424, buf423, buf420, buf416, buf415, buf411, buf408, buf406, buf403, buf399, buf398, buf394, buf391, buf389, buf386, buf382, buf381, buf378, buf374, buf373, buf369, buf366, buf364, buf360, buf357, buf355, buf351, buf348, buf346, buf343, buf339, buf338, buf335, buf331, buf330, buf326, buf323, buf321, buf318, buf314, buf313, buf309, buf306, buf304, buf301, buf297, buf296, buf293, buf289, buf288, buf284, buf281, buf279, buf275, buf272, buf270, buf266, buf263, buf261, buf258, buf254, buf253, buf250, buf246, buf245, buf241, buf238, buf236, buf233, buf229, buf228, buf224, buf221, buf219, buf216, buf212, buf211, buf208, buf204, buf203, buf199, buf196, buf194, buf190, buf187, buf185, buf181, buf178, buf176, buf173, buf169, buf168, buf165, buf161, buf160, buf156, buf153, buf151, buf148, buf144, buf143, buf139, buf136, buf134, buf130, buf127, buf125, buf121, buf118, buf116, buf112, buf109, buf106, buf102, buf92, buf99, buf95, buf92, buf89, buf85, buf82, buf80, buf77, buf73, buf72, buf69, buf65, buf64, buf60, buf57, buf55, buf51, buf47, buf45, buf41, buf38, buf36, buf33, buf29, buf28, buf25, buf21, buf20, buf16, buf13, buf11, buf7, buf4, reinterpret_tensor(buf1, (1000, 2048), (2048, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((104, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((104, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((104, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((26, 26, 3, 3), (234, 1, 78, 26), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((208, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((208, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((208, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((208, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((52, 52, 3, 3), (468, 1, 156, 52), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((416, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((104, 104, 3, 3), (936, 1, 312, 104), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((832, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((2048, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((832, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((2048, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((832, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((208, 208, 3, 3), (1872, 1, 624, 208), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((2048, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1023 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.int64)
    convolution_1 = rand_strided((8, 104, 56, 56), (326144, 1, 5824, 104), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_10 = rand_strided((8, 26, 56, 56), (326144, 1, 5824, 104), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((8, 26, 56, 56), (326144, 1, 5824, 104), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_24 = rand_strided((8, 26, 56, 56), (326144, 1, 5824, 104), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_31 = rand_strided((8, 26, 56, 56), (326144, 1, 5824, 104), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((8, 104, 56, 56), (326144, 1, 5824, 104), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 104, 56, 56), (326144, 1, 5824, 104), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_42 = rand_strided((8, 26, 56, 56), (326144, 1, 5824, 104), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_46 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_52 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((8, 104, 56, 56), (326144, 1, 5824, 104), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 104, 56, 56), (326144, 1, 5824, 104), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_72 = rand_strided((8, 26, 56, 56), (326144, 1, 5824, 104), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_74 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_80 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((8, 104, 56, 56), (326144, 1, 5824, 104), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 208, 56, 56), (652288, 1, 11648, 208), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_102 = rand_strided((8, 52, 56, 56), (652288, 1, 11648, 208), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_109 = rand_strided((8, 52, 56, 56), (652288, 1, 11648, 208), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_116 = rand_strided((8, 52, 56, 56), (652288, 1, 11648, 208), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_123 = rand_strided((8, 52, 56, 56), (652288, 1, 11648, 208), device='cuda:0', dtype=torch.float32)
    cat_3 = rand_strided((8, 208, 28, 28), (163072, 1, 5824, 208), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 208, 28, 28), (163072, 1, 5824, 208), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_134 = rand_strided((8, 52, 28, 28), (163072, 1, 5824, 208), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_133 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_139 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_4 = rand_strided((8, 208, 28, 28), (163072, 1, 5824, 208), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_25 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 208, 28, 28), (163072, 1, 5824, 208), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_164 = rand_strided((8, 52, 28, 28), (163072, 1, 5824, 208), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_161 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_167 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_5 = rand_strided((8, 208, 28, 28), (163072, 1, 5824, 208), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 208, 28, 28), (163072, 1, 5824, 208), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_194 = rand_strided((8, 52, 28, 28), (163072, 1, 5824, 208), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_189 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_195 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_6 = rand_strided((8, 208, 28, 28), (163072, 1, 5824, 208), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_35 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 416, 28, 28), (326144, 1, 11648, 416), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_224 = rand_strided((8, 104, 28, 28), (326144, 1, 11648, 416), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_231 = rand_strided((8, 104, 28, 28), (326144, 1, 11648, 416), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_238 = rand_strided((8, 104, 28, 28), (326144, 1, 11648, 416), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_245 = rand_strided((8, 104, 28, 28), (326144, 1, 11648, 416), device='cuda:0', dtype=torch.float32)
    cat_7 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_40 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_256 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_248 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_254 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_8 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_45 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_286 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_276 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_282 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_9 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_160 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_50 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_163 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_316 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_166 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_304 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_169 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_310 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_172 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_10 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_175 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_55 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_59 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_178 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_346 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_181 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_332 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_184 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_338 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_187 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_11 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_63 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_190 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_60 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_64 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_193 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_376 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_65 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_196 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_360 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_66 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_199 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_366 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_67 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_202 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_12 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_68 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_205 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_65 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_69 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_208 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_406 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_70 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_211 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_388 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_71 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_214 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_394 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_72 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_217 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_13 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_73 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_220 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_70 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_74 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_223 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_436 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_75 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_226 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_416 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_76 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_229 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_422 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_77 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_232 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_14 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_78 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_235 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_75 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_79 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_238 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_466 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_80 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_241 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_444 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_81 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_244 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_450 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_82 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_247 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_15 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_83 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_250 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_80 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_84 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_253 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_496 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_85 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_256 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_472 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_86 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_259 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_478 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_87 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_262 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_16 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_88 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_265 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_85 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_89 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_268 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_526 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_90 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_271 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_500 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_91 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_274 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_506 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_92 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_277 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_17 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_93 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_280 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_90 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_94 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_283 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_556 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_95 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_286 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_528 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_96 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_289 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_534 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_97 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_292 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_18 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_98 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_295 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_95 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_99 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_298 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_586 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_100 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_301 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_556 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_101 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_304 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_562 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_102 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_307 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_19 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_103 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_310 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_100 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_104 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_313 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_616 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_105 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_316 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_584 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_106 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_319 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_590 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_107 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_322 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_20 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_108 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_325 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_105 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_109 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_328 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_646 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_110 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_331 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_612 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_111 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_334 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_618 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_112 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_337 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_21 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_113 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_340 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_110 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_114 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_343 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_676 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_115 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_346 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_640 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_116 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_349 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_646 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_117 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_352 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_22 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_118 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_355 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_115 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_119 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_358 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_706 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_120 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_361 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_668 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_121 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_364 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_674 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_122 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_367 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_23 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_123 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_370 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_120 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_124 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_373 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_736 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_125 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_376 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_696 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_126 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_379 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_702 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_127 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_382 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_24 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_128 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_385 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_125 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_129 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_388 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_766 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_130 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_391 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_724 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_131 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_394 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_730 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_132 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_397 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_25 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_133 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_400 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_130 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_134 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_403 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_796 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_135 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_406 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_752 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_136 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_409 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_758 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_137 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_412 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_26 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_138 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_415 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_135 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_139 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_418 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_826 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_140 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_421 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_780 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_141 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_424 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_786 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_142 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_427 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_27 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_143 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_430 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_140 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_144 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_433 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_856 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_145 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_436 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_808 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_146 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_439 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_814 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_147 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_442 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_28 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_148 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_445 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_145 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_149 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    squeeze_448 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_886 = rand_strided((8, 104, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_150 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_451 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_836 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_151 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_454 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_842 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_152 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_457 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_29 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.float32)
    convolution_153 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_460 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_150 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_154 = rand_strided((8, 832, 14, 14), (163072, 1, 11648, 832), device='cuda:0', dtype=torch.float32)
    squeeze_463 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_916 = rand_strided((8, 208, 14, 14), (163072, 1, 11648, 832), device='cuda:0', dtype=torch.float32)
    convolution_155 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.float32)
    squeeze_466 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_923 = rand_strided((8, 208, 14, 14), (163072, 1, 11648, 832), device='cuda:0', dtype=torch.float32)
    convolution_156 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.float32)
    squeeze_469 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_930 = rand_strided((8, 208, 14, 14), (163072, 1, 11648, 832), device='cuda:0', dtype=torch.float32)
    convolution_157 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.float32)
    squeeze_472 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_937 = rand_strided((8, 208, 14, 14), (163072, 1, 11648, 832), device='cuda:0', dtype=torch.float32)
    cat_30 = rand_strided((8, 832, 7, 7), (40768, 1, 5824, 832), device='cuda:0', dtype=torch.float32)
    convolution_158 = rand_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    squeeze_475 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_159 = rand_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    squeeze_478 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_155 = rand_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    convolution_160 = rand_strided((8, 832, 7, 7), (40768, 1, 5824, 832), device='cuda:0', dtype=torch.float32)
    squeeze_481 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_948 = rand_strided((8, 208, 7, 7), (40768, 1, 5824, 832), device='cuda:0', dtype=torch.float32)
    convolution_161 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.float32)
    squeeze_484 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_895 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.float32)
    convolution_162 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.float32)
    squeeze_487 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_901 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.float32)
    convolution_163 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.float32)
    squeeze_490 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_31 = rand_strided((8, 832, 7, 7), (40768, 1, 5824, 832), device='cuda:0', dtype=torch.float32)
    convolution_164 = rand_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    squeeze_493 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_160 = rand_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    convolution_165 = rand_strided((8, 832, 7, 7), (40768, 1, 5824, 832), device='cuda:0', dtype=torch.float32)
    squeeze_496 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_978 = rand_strided((8, 208, 7, 7), (40768, 1, 5824, 832), device='cuda:0', dtype=torch.float32)
    convolution_166 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.float32)
    squeeze_499 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_923 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.float32)
    convolution_167 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.float32)
    squeeze_502 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_929 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.float32)
    convolution_168 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.float32)
    squeeze_505 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_32 = rand_strided((8, 832, 7, 7), (40768, 1, 5824, 832), device='cuda:0', dtype=torch.float32)
    convolution_169 = rand_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    squeeze_508 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((8, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.bool)
    unsqueeze_682 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_1 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.bool)
    unsqueeze_694 = rand_strided((1, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_2 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.bool)
    unsqueeze_706 = rand_strided((1, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_3 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.bool)
    unsqueeze_718 = rand_strided((1, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_4 = rand_strided((8, 832, 7, 7), (40768, 1, 5824, 832), device='cuda:0', dtype=torch.bool)
    unsqueeze_730 = rand_strided((1, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_742 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_6 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.bool)
    unsqueeze_754 = rand_strided((1, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_7 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.bool)
    unsqueeze_766 = rand_strided((1, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_8 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.bool)
    unsqueeze_778 = rand_strided((1, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_9 = rand_strided((8, 832, 7, 7), (40768, 1, 5824, 832), device='cuda:0', dtype=torch.bool)
    unsqueeze_790 = rand_strided((1, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_802 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_814 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_11 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.bool)
    unsqueeze_826 = rand_strided((1, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_12 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.bool)
    unsqueeze_838 = rand_strided((1, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_13 = rand_strided((8, 208, 7, 7), (10192, 1, 1456, 208), device='cuda:0', dtype=torch.bool)
    unsqueeze_850 = rand_strided((1, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_14 = rand_strided((8, 832, 14, 14), (163072, 1, 11648, 832), device='cuda:0', dtype=torch.bool)
    unsqueeze_862 = rand_strided((1, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_874 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_16 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_886 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_17 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_898 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_18 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_910 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_19 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_922 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_934 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_21 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_946 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_22 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_958 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_23 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_970 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_24 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_982 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_994 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_26 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1006 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_27 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1018 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_28 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1030 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_29 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_1042 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1054 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_31 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1066 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_32 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1078 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_33 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1090 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_34 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_1102 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1114 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_36 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1126 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_37 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1138 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_38 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1150 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_39 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_1162 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1174 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_41 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1186 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_42 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1198 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_43 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1210 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_44 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_1222 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1234 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_46 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1246 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_47 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1258 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_48 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1270 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_49 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_1282 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1294 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_51 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1306 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_52 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1318 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_53 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1330 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_54 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_1342 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1354 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_56 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1366 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_57 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1378 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_58 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1390 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_59 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_1402 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1414 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_61 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1426 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_62 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1438 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_63 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1450 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_64 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_1462 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1474 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_66 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1486 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_67 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1498 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_68 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1510 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_69 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_1522 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1534 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_71 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1546 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_72 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1558 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_73 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1570 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_74 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_1582 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1594 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_76 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1606 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_77 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1618 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_78 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1630 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_79 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_1642 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1654 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_81 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1666 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_82 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1678 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_83 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1690 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_84 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_1702 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1714 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_86 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1726 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_87 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1738 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_88 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1750 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_89 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_1762 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1774 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_91 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1786 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_92 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1798 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_93 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1810 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_94 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_1822 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1834 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_96 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1846 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_97 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1858 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_98 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1870 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_99 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_1882 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1894 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_101 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1906 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_102 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1918 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_103 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1930 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_104 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_1942 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_1954 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_106 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1966 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_107 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1978 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_108 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_1990 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_109 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_2002 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2014 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_111 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_2026 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_112 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_2038 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_113 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_2050 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_114 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_2062 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2074 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_116 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_2086 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_117 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_2098 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_118 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_2110 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_119 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_2122 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2134 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_121 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_2146 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_122 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_2158 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_123 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_2170 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_124 = rand_strided((8, 416, 14, 14), (81536, 1, 5824, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_2182 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2194 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2206 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_126 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_2218 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_127 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_2230 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_128 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_2242 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_129 = rand_strided((8, 416, 28, 28), (326144, 1, 11648, 416), device='cuda:0', dtype=torch.bool)
    unsqueeze_2254 = rand_strided((1, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2266 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_131 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.bool)
    unsqueeze_2278 = rand_strided((1, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_132 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.bool)
    unsqueeze_2290 = rand_strided((1, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_133 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.bool)
    unsqueeze_2302 = rand_strided((1, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_134 = rand_strided((8, 208, 28, 28), (163072, 1, 5824, 208), device='cuda:0', dtype=torch.bool)
    unsqueeze_2314 = rand_strided((1, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2326 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_136 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.bool)
    unsqueeze_2338 = rand_strided((1, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_137 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.bool)
    unsqueeze_2350 = rand_strided((1, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_138 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.bool)
    unsqueeze_2362 = rand_strided((1, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_139 = rand_strided((8, 208, 28, 28), (163072, 1, 5824, 208), device='cuda:0', dtype=torch.bool)
    unsqueeze_2374 = rand_strided((1, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2386 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_141 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.bool)
    unsqueeze_2398 = rand_strided((1, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_142 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.bool)
    unsqueeze_2410 = rand_strided((1, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_143 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.bool)
    unsqueeze_2422 = rand_strided((1, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_144 = rand_strided((8, 208, 28, 28), (163072, 1, 5824, 208), device='cuda:0', dtype=torch.bool)
    unsqueeze_2434 = rand_strided((1, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2446 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2458 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_146 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.bool)
    unsqueeze_2470 = rand_strided((1, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_147 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.bool)
    unsqueeze_2482 = rand_strided((1, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_148 = rand_strided((8, 52, 28, 28), (40768, 1, 1456, 52), device='cuda:0', dtype=torch.bool)
    unsqueeze_2494 = rand_strided((1, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_149 = rand_strided((8, 208, 56, 56), (652288, 1, 11648, 208), device='cuda:0', dtype=torch.bool)
    unsqueeze_2506 = rand_strided((1, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2518 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_151 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.bool)
    unsqueeze_2530 = rand_strided((1, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_152 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.bool)
    unsqueeze_2542 = rand_strided((1, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_153 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.bool)
    unsqueeze_2554 = rand_strided((1, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_154 = rand_strided((8, 104, 56, 56), (326144, 1, 5824, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_2566 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2578 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_156 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.bool)
    unsqueeze_2590 = rand_strided((1, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_157 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.bool)
    unsqueeze_2602 = rand_strided((1, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_158 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.bool)
    unsqueeze_2614 = rand_strided((1, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_159 = rand_strided((8, 104, 56, 56), (326144, 1, 5824, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_2626 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2638 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2650 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_161 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.bool)
    unsqueeze_2662 = rand_strided((1, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_162 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.bool)
    unsqueeze_2674 = rand_strided((1, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_163 = rand_strided((8, 26, 56, 56), (81536, 1, 1456, 26), device='cuda:0', dtype=torch.bool)
    unsqueeze_2686 = rand_strided((1, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_164 = rand_strided((8, 104, 56, 56), (326144, 1, 5824, 104), device='cuda:0', dtype=torch.bool)
    unsqueeze_2698 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2710 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_281, primals_283, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_296, primals_298, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_311, primals_313, primals_314, primals_316, primals_317, primals_319, primals_320, primals_322, primals_323, primals_325, primals_326, primals_328, primals_329, primals_331, primals_332, primals_334, primals_335, primals_337, primals_338, primals_340, primals_341, primals_343, primals_344, primals_346, primals_347, primals_349, primals_350, primals_352, primals_353, primals_355, primals_356, primals_358, primals_359, primals_361, primals_362, primals_364, primals_365, primals_367, primals_368, primals_370, primals_371, primals_373, primals_374, primals_376, primals_377, primals_379, primals_380, primals_382, primals_383, primals_385, primals_386, primals_388, primals_389, primals_391, primals_392, primals_394, primals_395, primals_397, primals_398, primals_400, primals_401, primals_403, primals_404, primals_406, primals_407, primals_409, primals_410, primals_412, primals_413, primals_415, primals_416, primals_418, primals_419, primals_421, primals_422, primals_424, primals_425, primals_427, primals_428, primals_430, primals_431, primals_433, primals_434, primals_436, primals_437, primals_439, primals_440, primals_442, primals_443, primals_445, primals_446, primals_448, primals_449, primals_451, primals_452, primals_454, primals_455, primals_457, primals_458, primals_460, primals_461, primals_463, primals_464, primals_466, primals_467, primals_469, primals_470, primals_472, primals_473, primals_475, primals_476, primals_478, primals_479, primals_481, primals_482, primals_484, primals_485, primals_487, primals_488, primals_490, primals_491, primals_493, primals_494, primals_496, primals_497, primals_499, primals_500, primals_502, primals_503, primals_505, primals_506, primals_508, primals_509, primals_1023, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, getitem_10, convolution_2, squeeze_7, getitem_17, convolution_3, squeeze_10, getitem_24, convolution_4, squeeze_13, getitem_31, cat, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, getitem_42, convolution_8, squeeze_25, add_46, convolution_9, squeeze_28, add_52, convolution_10, squeeze_31, cat_1, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, getitem_72, convolution_13, squeeze_40, add_74, convolution_14, squeeze_43, add_80, convolution_15, squeeze_46, cat_2, convolution_16, squeeze_49, relu_15, convolution_17, squeeze_52, getitem_102, convolution_18, squeeze_55, getitem_109, convolution_19, squeeze_58, getitem_116, convolution_20, squeeze_61, getitem_123, cat_3, convolution_21, squeeze_64, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, getitem_134, convolution_24, squeeze_73, add_133, convolution_25, squeeze_76, add_139, convolution_26, squeeze_79, cat_4, convolution_27, squeeze_82, relu_25, convolution_28, squeeze_85, getitem_164, convolution_29, squeeze_88, add_161, convolution_30, squeeze_91, add_167, convolution_31, squeeze_94, cat_5, convolution_32, squeeze_97, relu_30, convolution_33, squeeze_100, getitem_194, convolution_34, squeeze_103, add_189, convolution_35, squeeze_106, add_195, convolution_36, squeeze_109, cat_6, convolution_37, squeeze_112, relu_35, convolution_38, squeeze_115, getitem_224, convolution_39, squeeze_118, getitem_231, convolution_40, squeeze_121, getitem_238, convolution_41, squeeze_124, getitem_245, cat_7, convolution_42, squeeze_127, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, getitem_256, convolution_45, squeeze_136, add_248, convolution_46, squeeze_139, add_254, convolution_47, squeeze_142, cat_8, convolution_48, squeeze_145, relu_45, convolution_49, squeeze_148, getitem_286, convolution_50, squeeze_151, add_276, convolution_51, squeeze_154, add_282, convolution_52, squeeze_157, cat_9, convolution_53, squeeze_160, relu_50, convolution_54, squeeze_163, getitem_316, convolution_55, squeeze_166, add_304, convolution_56, squeeze_169, add_310, convolution_57, squeeze_172, cat_10, convolution_58, squeeze_175, relu_55, convolution_59, squeeze_178, getitem_346, convolution_60, squeeze_181, add_332, convolution_61, squeeze_184, add_338, convolution_62, squeeze_187, cat_11, convolution_63, squeeze_190, relu_60, convolution_64, squeeze_193, getitem_376, convolution_65, squeeze_196, add_360, convolution_66, squeeze_199, add_366, convolution_67, squeeze_202, cat_12, convolution_68, squeeze_205, relu_65, convolution_69, squeeze_208, getitem_406, convolution_70, squeeze_211, add_388, convolution_71, squeeze_214, add_394, convolution_72, squeeze_217, cat_13, convolution_73, squeeze_220, relu_70, convolution_74, squeeze_223, getitem_436, convolution_75, squeeze_226, add_416, convolution_76, squeeze_229, add_422, convolution_77, squeeze_232, cat_14, convolution_78, squeeze_235, relu_75, convolution_79, squeeze_238, getitem_466, convolution_80, squeeze_241, add_444, convolution_81, squeeze_244, add_450, convolution_82, squeeze_247, cat_15, convolution_83, squeeze_250, relu_80, convolution_84, squeeze_253, getitem_496, convolution_85, squeeze_256, add_472, convolution_86, squeeze_259, add_478, convolution_87, squeeze_262, cat_16, convolution_88, squeeze_265, relu_85, convolution_89, squeeze_268, getitem_526, convolution_90, squeeze_271, add_500, convolution_91, squeeze_274, add_506, convolution_92, squeeze_277, cat_17, convolution_93, squeeze_280, relu_90, convolution_94, squeeze_283, getitem_556, convolution_95, squeeze_286, add_528, convolution_96, squeeze_289, add_534, convolution_97, squeeze_292, cat_18, convolution_98, squeeze_295, relu_95, convolution_99, squeeze_298, getitem_586, convolution_100, squeeze_301, add_556, convolution_101, squeeze_304, add_562, convolution_102, squeeze_307, cat_19, convolution_103, squeeze_310, relu_100, convolution_104, squeeze_313, getitem_616, convolution_105, squeeze_316, add_584, convolution_106, squeeze_319, add_590, convolution_107, squeeze_322, cat_20, convolution_108, squeeze_325, relu_105, convolution_109, squeeze_328, getitem_646, convolution_110, squeeze_331, add_612, convolution_111, squeeze_334, add_618, convolution_112, squeeze_337, cat_21, convolution_113, squeeze_340, relu_110, convolution_114, squeeze_343, getitem_676, convolution_115, squeeze_346, add_640, convolution_116, squeeze_349, add_646, convolution_117, squeeze_352, cat_22, convolution_118, squeeze_355, relu_115, convolution_119, squeeze_358, getitem_706, convolution_120, squeeze_361, add_668, convolution_121, squeeze_364, add_674, convolution_122, squeeze_367, cat_23, convolution_123, squeeze_370, relu_120, convolution_124, squeeze_373, getitem_736, convolution_125, squeeze_376, add_696, convolution_126, squeeze_379, add_702, convolution_127, squeeze_382, cat_24, convolution_128, squeeze_385, relu_125, convolution_129, squeeze_388, getitem_766, convolution_130, squeeze_391, add_724, convolution_131, squeeze_394, add_730, convolution_132, squeeze_397, cat_25, convolution_133, squeeze_400, relu_130, convolution_134, squeeze_403, getitem_796, convolution_135, squeeze_406, add_752, convolution_136, squeeze_409, add_758, convolution_137, squeeze_412, cat_26, convolution_138, squeeze_415, relu_135, convolution_139, squeeze_418, getitem_826, convolution_140, squeeze_421, add_780, convolution_141, squeeze_424, add_786, convolution_142, squeeze_427, cat_27, convolution_143, squeeze_430, relu_140, convolution_144, squeeze_433, getitem_856, convolution_145, squeeze_436, add_808, convolution_146, squeeze_439, add_814, convolution_147, squeeze_442, cat_28, convolution_148, squeeze_445, relu_145, convolution_149, squeeze_448, getitem_886, convolution_150, squeeze_451, add_836, convolution_151, squeeze_454, add_842, convolution_152, squeeze_457, cat_29, convolution_153, squeeze_460, relu_150, convolution_154, squeeze_463, getitem_916, convolution_155, squeeze_466, getitem_923, convolution_156, squeeze_469, getitem_930, convolution_157, squeeze_472, getitem_937, cat_30, convolution_158, squeeze_475, convolution_159, squeeze_478, relu_155, convolution_160, squeeze_481, getitem_948, convolution_161, squeeze_484, add_895, convolution_162, squeeze_487, add_901, convolution_163, squeeze_490, cat_31, convolution_164, squeeze_493, relu_160, convolution_165, squeeze_496, getitem_978, convolution_166, squeeze_499, add_923, convolution_167, squeeze_502, add_929, convolution_168, squeeze_505, cat_32, convolution_169, squeeze_508, view, permute_1, le, unsqueeze_682, le_1, unsqueeze_694, le_2, unsqueeze_706, le_3, unsqueeze_718, le_4, unsqueeze_730, unsqueeze_742, le_6, unsqueeze_754, le_7, unsqueeze_766, le_8, unsqueeze_778, le_9, unsqueeze_790, unsqueeze_802, unsqueeze_814, le_11, unsqueeze_826, le_12, unsqueeze_838, le_13, unsqueeze_850, le_14, unsqueeze_862, unsqueeze_874, le_16, unsqueeze_886, le_17, unsqueeze_898, le_18, unsqueeze_910, le_19, unsqueeze_922, unsqueeze_934, le_21, unsqueeze_946, le_22, unsqueeze_958, le_23, unsqueeze_970, le_24, unsqueeze_982, unsqueeze_994, le_26, unsqueeze_1006, le_27, unsqueeze_1018, le_28, unsqueeze_1030, le_29, unsqueeze_1042, unsqueeze_1054, le_31, unsqueeze_1066, le_32, unsqueeze_1078, le_33, unsqueeze_1090, le_34, unsqueeze_1102, unsqueeze_1114, le_36, unsqueeze_1126, le_37, unsqueeze_1138, le_38, unsqueeze_1150, le_39, unsqueeze_1162, unsqueeze_1174, le_41, unsqueeze_1186, le_42, unsqueeze_1198, le_43, unsqueeze_1210, le_44, unsqueeze_1222, unsqueeze_1234, le_46, unsqueeze_1246, le_47, unsqueeze_1258, le_48, unsqueeze_1270, le_49, unsqueeze_1282, unsqueeze_1294, le_51, unsqueeze_1306, le_52, unsqueeze_1318, le_53, unsqueeze_1330, le_54, unsqueeze_1342, unsqueeze_1354, le_56, unsqueeze_1366, le_57, unsqueeze_1378, le_58, unsqueeze_1390, le_59, unsqueeze_1402, unsqueeze_1414, le_61, unsqueeze_1426, le_62, unsqueeze_1438, le_63, unsqueeze_1450, le_64, unsqueeze_1462, unsqueeze_1474, le_66, unsqueeze_1486, le_67, unsqueeze_1498, le_68, unsqueeze_1510, le_69, unsqueeze_1522, unsqueeze_1534, le_71, unsqueeze_1546, le_72, unsqueeze_1558, le_73, unsqueeze_1570, le_74, unsqueeze_1582, unsqueeze_1594, le_76, unsqueeze_1606, le_77, unsqueeze_1618, le_78, unsqueeze_1630, le_79, unsqueeze_1642, unsqueeze_1654, le_81, unsqueeze_1666, le_82, unsqueeze_1678, le_83, unsqueeze_1690, le_84, unsqueeze_1702, unsqueeze_1714, le_86, unsqueeze_1726, le_87, unsqueeze_1738, le_88, unsqueeze_1750, le_89, unsqueeze_1762, unsqueeze_1774, le_91, unsqueeze_1786, le_92, unsqueeze_1798, le_93, unsqueeze_1810, le_94, unsqueeze_1822, unsqueeze_1834, le_96, unsqueeze_1846, le_97, unsqueeze_1858, le_98, unsqueeze_1870, le_99, unsqueeze_1882, unsqueeze_1894, le_101, unsqueeze_1906, le_102, unsqueeze_1918, le_103, unsqueeze_1930, le_104, unsqueeze_1942, unsqueeze_1954, le_106, unsqueeze_1966, le_107, unsqueeze_1978, le_108, unsqueeze_1990, le_109, unsqueeze_2002, unsqueeze_2014, le_111, unsqueeze_2026, le_112, unsqueeze_2038, le_113, unsqueeze_2050, le_114, unsqueeze_2062, unsqueeze_2074, le_116, unsqueeze_2086, le_117, unsqueeze_2098, le_118, unsqueeze_2110, le_119, unsqueeze_2122, unsqueeze_2134, le_121, unsqueeze_2146, le_122, unsqueeze_2158, le_123, unsqueeze_2170, le_124, unsqueeze_2182, unsqueeze_2194, unsqueeze_2206, le_126, unsqueeze_2218, le_127, unsqueeze_2230, le_128, unsqueeze_2242, le_129, unsqueeze_2254, unsqueeze_2266, le_131, unsqueeze_2278, le_132, unsqueeze_2290, le_133, unsqueeze_2302, le_134, unsqueeze_2314, unsqueeze_2326, le_136, unsqueeze_2338, le_137, unsqueeze_2350, le_138, unsqueeze_2362, le_139, unsqueeze_2374, unsqueeze_2386, le_141, unsqueeze_2398, le_142, unsqueeze_2410, le_143, unsqueeze_2422, le_144, unsqueeze_2434, unsqueeze_2446, unsqueeze_2458, le_146, unsqueeze_2470, le_147, unsqueeze_2482, le_148, unsqueeze_2494, le_149, unsqueeze_2506, unsqueeze_2518, le_151, unsqueeze_2530, le_152, unsqueeze_2542, le_153, unsqueeze_2554, le_154, unsqueeze_2566, unsqueeze_2578, le_156, unsqueeze_2590, le_157, unsqueeze_2602, le_158, unsqueeze_2614, le_159, unsqueeze_2626, unsqueeze_2638, unsqueeze_2650, le_161, unsqueeze_2662, le_162, unsqueeze_2674, le_163, unsqueeze_2686, le_164, unsqueeze_2698, unsqueeze_2710, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('res2net101_26w_4s', benchmark_compiled_module)
